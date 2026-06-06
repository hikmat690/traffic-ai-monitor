"""
Core detection engine.
Handles: vehicle detection, multi-object tracking, zone counting, parking slot logic.
"""

import cv2
import numpy as np
import time
import threading
from collections import defaultdict, deque
from datetime import datetime

import supervision as sv
from ultralytics import YOLO

from utils.models import Session, VehicleCount, ParkingSnapshot
from utils.alerts import send_alert

# COCO class IDs we care about
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
CONFIDENCE      = float(__import__("os").getenv("CONFIDENCE_THRESHOLD", 0.45))
IOU_THRESH      = float(__import__("os").getenv("IOU_THRESHOLD", 0.5))
ALERT_PCT       = float(__import__("os").getenv("ALERT_OCCUPANCY_PERCENT", 85))


class Zone:
    """Represents a counting or parking zone defined by a polygon."""

    def __init__(self, name: str, polygon: np.ndarray, zone_type: str = "counting"):
        self.name       = name
        self.polygon    = polygon          # (N,2) int array
        self.zone_type  = zone_type        # "counting" | "parking"
        self.color      = (0, 200, 100)
        self.count      = 0               # vehicles currently inside
        self.total_in   = 0               # cumulative entries
        self.occupied   = 0               # parking slots taken
        self.total_slots = 0              # parking total slots
        self.track_ids_inside: set = set()
        self.last_alert_time    = 0

    @property
    def occupancy_pct(self):
        if self.total_slots == 0:
            return 0
        return round(100 * self.occupied / self.total_slots, 1)

    def sv_zone(self) -> sv.PolygonZone:
        return sv.PolygonZone(polygon=self.polygon)


class ParkingSlot:
    """Single parking slot polygon."""

    def __init__(self, slot_id: int, polygon: np.ndarray):
        self.slot_id  = slot_id
        self.polygon  = polygon
        self.occupied = False
        self.track_id = None

    def sv_zone(self) -> sv.PolygonZone:
        return sv.PolygonZone(polygon=self.polygon)


class DetectionEngine:
    """
    Main class that wraps YOLO + ByteTrack + zone logic.
    Call `start(source)` to begin processing in a background thread.
    Read `latest_frame` and `stats` for the dashboard.
    """

    def __init__(self, socketio=None):
        self.model     = YOLO("yolov8n.pt")   # auto-downloads first time
        self.tracker   = sv.ByteTrack()
        self.socketio  = socketio

        self.zones: list[Zone]         = []
        self.parking_slots: list[ParkingSlot] = []

        self.latest_frame: np.ndarray | None = None
        self.stats: dict = {}

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        # Speed estimation: track centroid history per ID
        self._centroid_history: dict[int, deque] = defaultdict(lambda: deque(maxlen=10))
        self._fps_history = deque(maxlen=30)

        # Throttle DB writes (every 30 s)
        self._last_db_write = 0

    # ── Zone setup ────────────────────────────────────────────────────────────

    def add_counting_zone(self, name: str, polygon: list):
        z = Zone(name, np.array(polygon, dtype=np.int32), "counting")
        self.zones.append(z)
        return z

    def add_parking_zone(self, name: str, slots: list[list]):
        """
        slots: list of polygons, each polygon = list of [x,y] points.
        """
        zone = Zone(name, np.array(slots[0], dtype=np.int32), "parking")
        zone.total_slots = len(slots)
        self.zones.append(zone)
        for i, poly in enumerate(slots):
            self.parking_slots.append(ParkingSlot(i, np.array(poly, dtype=np.int32)))
        return zone

    # ── Source helpers ────────────────────────────────────────────────────────

    @staticmethod
    def resolve_source(source: str | int):
        """Accept webcam index, file path, RTSP URL, or YouTube URL."""
        if isinstance(source, int):
            return source
        if source.isdigit():
            return int(source)
        if "youtube.com" in source or "youtu.be" in source:
            try:
                import yt_dlp
                ydl_opts = {"quiet": True, "format": "best[height<=720]"}
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(source, download=False)
                    url  = info.get("url") or info["formats"][-1]["url"]
                    print(f"[Source] YouTube stream resolved: {url[:80]}…")
                    return url
            except Exception as e:
                print(f"[Source] yt-dlp error: {e}. Falling back to direct URL.")
        return source

    # ── Main loop ─────────────────────────────────────────────────────────────

    def start(self, source):
        self._stop_event.clear()
        resolved = self.resolve_source(source)
        self._thread = threading.Thread(
            target=self._run, args=(resolved,), daemon=True
        )
        self._thread.start()

    def stop(self):
        self._stop_event.set()

    def _run(self, source):
        if isinstance(source, str) and source.startswith("http"):
            cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        else:
            cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"[Engine] Cannot open source: {source}")
            return

        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[Engine] Stream opened — {frame_w}x{frame_h}")

        # Build supervision zone objects once
        sv_zones    = [z.sv_zone() for z in self.zones if z.zone_type == "counting"]
        sv_pk_zones = [s.sv_zone() for s in self.parking_slots]

        # Annotators
        box_ann    = sv.BoxAnnotator(thickness=2)
        label_ann  = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)
        trace_ann  = sv.TraceAnnotator(thickness=2, trace_length=30)
        zone_anns  = [sv.PolygonZoneAnnotator(zone=z, color=sv.Color(0, 200, 100), thickness=2)
                      for z in sv_zones]

        prev_time = time.time()

        while not self._stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                # Loop video files; end for live streams
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            t0 = time.time()

            # ── Detection ────────────────────────────────────────────────────
            results = self.model(
                frame,
                conf=CONFIDENCE,
                iou=IOU_THRESH,
                classes=list(VEHICLE_CLASSES.keys()),
                verbose=False,
            )[0]

            detections = sv.Detections.from_ultralytics(results)
            detections = self.tracker.update_with_detections(detections)

            # ── Speed estimation (pixels/s → rough km/h) ─────────────────────
            speed_map: dict[int, float] = {}
            for det_idx in range(len(detections)):
                if (
                    detections.tracker_id is not None
                     and len(detections.tracker_id) > det_idx
                    ):
                    tid = int(detections.tracker_id[det_idx])
                else:
                 tid = det_idx
                cx  = int((detections.xyxy[det_idx][0] + detections.xyxy[det_idx][2]) / 2)
                cy  = int((detections.xyxy[det_idx][1] + detections.xyxy[det_idx][3]) / 2)
                self._centroid_history[tid].append((cx, cy, time.time()))
                if len(self._centroid_history[tid]) >= 2:
                    (x1, y1, t1), (x2, y2, t2) = (
                        self._centroid_history[tid][0],
                        self._centroid_history[tid][-1],
                    )
                    dt = t2 - t1
                    if dt > 0:
                        px_s = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5 / dt
                        # Rough px→km/h: assume 1 px ≈ 0.05 m at typical camera height
                        speed_map[tid] = round(px_s * 0.05 * 3.6, 1)

            # ── Zone counting ─────────────────────────────────────────────────
            zone_stats = {}
            for i, (zone, sv_z) in enumerate(zip(
                [z for z in self.zones if z.zone_type == "counting"], sv_zones
            )):
                

                if detections.tracker_id is None or len(detections.tracker_id) == 0:
                 detections.tracker_id = np.full(len(detections), -1)

                mask = sv_z.trigger(detections=detections)
                inside   = detections[mask]
                ids_now  = set(inside.tracker_id.tolist()) if inside.tracker_id is not None else set()
                new_ids  = ids_now - zone.track_ids_inside
                zone.total_in      += len(new_ids)
                zone.track_ids_inside = ids_now
                zone.count         = len(ids_now)

                # Type breakdown
                type_counts: dict[str, int] = defaultdict(int)
                if inside.class_id is not None:
                    for cid in inside.class_id:
                        type_counts[VEHICLE_CLASSES.get(int(cid), "vehicle")] += 1

                zone_stats[zone.name] = {
                    "current": zone.count,
                    "total_in": zone.total_in,
                    "types": dict(type_counts),
                }

            # ── Parking slots ─────────────────────────────────────────────────
            pk_zone_obj = next((z for z in self.zones if z.zone_type == "parking"), None)
            for slot, sv_pz in zip(self.parking_slots, sv_pk_zones):
                mask         = sv_pz.trigger(detections=detections)
                slot.occupied = bool(mask.any())
                if slot.occupied and detections.tracker_id is not None:
                    ids_in = detections.tracker_id[mask]
                    slot.track_id = int(ids_in[0]) if len(ids_in) else None
                else:
                    slot.track_id = None

            if pk_zone_obj:
                pk_zone_obj.occupied = sum(1 for s in self.parking_slots if s.occupied)

            # ── Alerts ────────────────────────────────────────────────────────
            now = time.time()
            for zone in self.zones:
                if zone.zone_type == "parking" and zone.total_slots:
                    pct = zone.occupancy_pct
                    if pct >= ALERT_PCT and (now - zone.last_alert_time) > 300:
                        zone.last_alert_time = now
                        send_alert(zone.name, f"Parking {pct:.0f}% full ({zone.occupied}/{zone.total_slots} slots)")

            # ── DB write (throttled) ──────────────────────────────────────────
            if now - self._last_db_write > 30:
                self._last_db_write = now
                self._write_db(zone_stats)

            # ── Annotate frame ────────────────────────────────────────────────
            annotated = frame.copy()

            # Draw parking slots
            for slot in self.parking_slots:
                color = (0, 60, 220) if slot.occupied else (0, 200, 80)
                cv2.polylines(annotated, [slot.polygon], True, color, 2)
                cx = int(slot.polygon[:, 0].mean())
                cy = int(slot.polygon[:, 1].mean())
                status = "OCC" if slot.occupied else "FREE"
                cv2.putText(annotated, status, (cx - 14, cy + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)

            # Draw counting zones
            for z_ann in zone_anns:
                annotated = z_ann.annotate(annotated)

            # Draw boxes + labels
            if len(detections):
                labels = []
                for idx in range(len(detections)):
                    tid   = int(detections.tracker_id[idx]) if detections.tracker_id is not None else 0
                    cid   = int(detections.class_id[idx])
                    conf  = float(detections.confidence[idx])
                    vtype = VEHICLE_CLASSES.get(cid, "vehicle")
                    spd   = speed_map.get(tid, 0)
                    labels.append(f"#{tid} {vtype} {conf:.0%} ~{spd}km/h")

                annotated = trace_ann.annotate(annotated, detections=detections)
                annotated = box_ann.annotate(annotated, detections=detections)
                annotated = label_ann.annotate(annotated, detections=detections, labels=labels)

            # FPS overlay
            fps = 1.0 / max(time.time() - t0, 1e-6)
            self._fps_history.append(fps)
            avg_fps = sum(self._fps_history) / len(self._fps_history)
            cv2.putText(annotated, f"FPS: {avg_fps:.1f}", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(annotated, datetime.now().strftime("%Y-%m-%d  %H:%M:%S"),
                        (10, frame_h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (200, 200, 200), 1, cv2.LINE_AA)

            # ── Publish ───────────────────────────────────────────────────────
            stats_payload = {
                "fps": round(avg_fps, 1),
                "zones": zone_stats,
                "parking": {
                    "slots": [
                        {"id": s.slot_id, "occupied": s.occupied} for s in self.parking_slots
                    ],
                    "summary": {
                        z.name: {
                            "total": z.total_slots,
                            "occupied": z.occupied,
                            "free": z.total_slots - z.occupied,
                            "pct": z.occupancy_pct,
                        }
                        for z in self.zones if z.zone_type == "parking"
                    },
                },
                "timestamp": datetime.now().isoformat(),
            }

            with self._lock:
                self.latest_frame = annotated.copy()
                self.stats = stats_payload

            if self.socketio:
                self.socketio.emit("stats_update", stats_payload)

        cap.release()
        print("[Engine] Stream closed.")

    def _write_db(self, zone_stats: dict):
        session = Session()
        try:
            for zname, zdata in zone_stats.items():
                for vtype, cnt in zdata.get("types", {}).items():
                    session.add(VehicleCount(zone_name=zname, count=cnt, vehicle_type=vtype))

            for zone in self.zones:
                if zone.zone_type == "parking" and zone.total_slots:
                    session.add(ParkingSnapshot(
                        zone_name=zone.name,
                        total_slots=zone.total_slots,
                        occupied=zone.occupied,
                        occupancy_pct=zone.occupancy_pct,
                    ))
            session.commit()
        except Exception as e:
            print(f"[DB] Write error: {e}")
            session.rollback()
        finally:
            session.close()

    # ── Frame streaming ───────────────────────────────────────────────────────

    def get_jpeg_frame(self) -> bytes | None:
        with self._lock:
            if self.latest_frame is None:
                return None
            _, buf = cv2.imencode(".jpg", self.latest_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            return buf.tobytes()