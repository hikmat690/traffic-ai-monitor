"""
Detection Engine — Fixed version
Fixes: 1) YouTube URL support  2) Fast startup (model loads in background thread)
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

VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
CONFIDENCE  = 0.45
IOU_THRESH  = 0.5
ALERT_PCT   = 85.0


class Zone:
    def __init__(self, name, polygon, zone_type="counting"):
        self.name            = name
        self.polygon         = polygon
        self.zone_type       = zone_type
        self.count           = 0
        self.total_in        = 0
        self.occupied        = 0
        self.total_slots     = 0
        self.track_ids_inside: set = set()
        self.last_alert_time = 0

    @property
    def occupancy_pct(self):
        if self.total_slots == 0:
            return 0
        return round(100 * self.occupied / self.total_slots, 1)

    def sv_zone(self):
        return sv.PolygonZone(polygon=self.polygon)


class ParkingSlot:
    def __init__(self, slot_id, polygon):
        self.slot_id  = slot_id
        self.polygon  = polygon
        self.occupied = False

    def sv_zone(self):
        return sv.PolygonZone(polygon=self.polygon)


class DetectionEngine:
    def __init__(self, socketio=None):
        self.socketio      = socketio
        self.zones         = []
        self.parking_slots = []
        self.latest_frame  = None
        self.stats         = {}
        self._thread       = None
        self._stop_event   = threading.Event()
        self._lock         = threading.Lock()
        self._fps_history  = deque(maxlen=30)
        self._centroid_history = defaultdict(lambda: deque(maxlen=10))
        self._last_db_write    = 0
        self._model            = None  # loaded lazily in background
        self._model_ready      = threading.Event()

        # Load model in background so app.py starts instantly
        threading.Thread(target=self._load_model, daemon=True).start()

    def _load_model(self):
        print("[Engine] Loading YOLOv8 model in background...")
        self._model = YOLO("yolov8n.pt")
        self._model_ready.set()
        print("[Engine] YOLOv8 model ready!")

    # ── Zone helpers ─────────────────────────────────────────────────────────

    def add_counting_zone(self, name, polygon):
        z = Zone(name, np.array(polygon, dtype=np.int32), "counting")
        self.zones.append(z)
        return z

    def add_parking_zone(self, name, slots):
        zone = Zone(name, np.array(slots[0], dtype=np.int32), "parking")
        zone.total_slots = len(slots)
        self.zones.append(zone)
        for i, poly in enumerate(slots):
            self.parking_slots.append(ParkingSlot(i, np.array(poly, dtype=np.int32)))
        return zone

    # ── YouTube URL resolver ──────────────────────────────────────────────────

    @staticmethod
    def resolve_youtube(url):
        """Extract direct video stream URL from a YouTube link using yt-dlp."""
        try:
            import yt_dlp
            ydl_opts = {
                "quiet": True,
                "no_warnings": True,
                # prefer mp4 at 480p or lower for speed
                "format": "best[height<=480][ext=mp4]/best[height<=480]/best",
            }
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                # get direct URL
                direct = info.get("url")
                if not direct:
                    fmts = info.get("formats", [])
                    # pick last (best) format that has a url
                    for f in reversed(fmts):
                        if f.get("url") and f.get("protocol") in ("https", "http", "m3u8_native", "m3u8"):
                            direct = f["url"]
                            break
                if direct:
                    print(f"[Engine] YouTube resolved OK")
                    return direct
        except Exception as e:
            print(f"[Engine] yt-dlp error: {e}")
        return None

    @staticmethod
    def resolve_source(source):
        if isinstance(source, int):
            return source
        s = str(source).strip()
        if s.isdigit():
            return int(s)
        if "youtube.com" in s or "youtu.be" in s:
            direct = DetectionEngine.resolve_youtube(s)
            if direct:
                return direct
            print("[Engine] Could not resolve YouTube URL.")
            return None
        return s  # file path or RTSP

    # ── Start / Stop ──────────────────────────────────────────────────────────

    def start(self, source):
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, args=(source,), daemon=True
        )
        self._thread.start()

    def stop(self):
        self._stop_event.set()

    # ── Main detection loop ───────────────────────────────────────────────────

    def _run(self, source):
        # Wait for model (max 30 s)
        if not self._model_ready.wait(timeout=30):
            print("[Engine] Model not ready after 30s — aborting.")
            return

        resolved = self.resolve_source(source)
        if resolved is None:
            print("[Engine] Cannot resolve source — aborting.")
            self._emit_error("Cannot open video source. Try a different URL or use 0 for webcam.")
            return

        # Open video
        if isinstance(resolved, str) and (resolved.startswith("http") or resolved.startswith("rtsp")):
            cap = cv2.VideoCapture(resolved, cv2.CAP_FFMPEG)
        else:
            cap = cv2.VideoCapture(resolved)

        if not cap.isOpened():
            print(f"[Engine] Cannot open: {resolved[:80]}")
            self._emit_error("Cannot open video source. Check the URL or try webcam (0).")
            return

        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[Engine] Stream opened — {frame_w}x{frame_h}")

        tracker   = sv.ByteTrack()
        box_ann   = sv.BoxAnnotator(thickness=2)
        label_ann = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)
        trace_ann = sv.TraceAnnotator(thickness=2, trace_length=25)

        sv_zones    = [z.sv_zone() for z in self.zones if z.zone_type == "counting"]
        sv_pk_zones = [s.sv_zone() for s in self.parking_slots]

        zone_anns = [
            sv.PolygonZoneAnnotator(zone=z, color=sv.Color(0, 200, 100), thickness=2)
            for z in sv_zones
        ]

        while not self._stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                time.sleep(0.1)
                continue

            t0 = time.time()

            # Detection
            results = self._model(
                frame,
                conf=CONFIDENCE,
                iou=IOU_THRESH,
                classes=list(VEHICLE_CLASSES.keys()),
                verbose=False,
            )[0]

            detections = sv.Detections.from_ultralytics(results)
            detections = tracker.update_with_detections(detections)

            # Speed estimation
            speed_map = {}
            for i in range(len(detections)):
                if detections.tracker_id is None:
                    break
                tid = int(detections.tracker_id[i])
                cx  = int((detections.xyxy[i][0] + detections.xyxy[i][2]) / 2)
                cy  = int((detections.xyxy[i][1] + detections.xyxy[i][3]) / 2)
                self._centroid_history[tid].append((cx, cy, time.time()))
                h = self._centroid_history[tid]
                if len(h) >= 2:
                    dt = h[-1][2] - h[0][2]
                    if dt > 0:
                        px_s = ((h[-1][0]-h[0][0])**2 + (h[-1][1]-h[0][1])**2)**0.5 / dt
                        speed_map[tid] = round(px_s * 0.05 * 3.6, 1)

            # Zone counting
            zone_stats = {}
            for zone, sv_z in zip([z for z in self.zones if z.zone_type=="counting"], sv_zones):
                mask    = sv_z.trigger(detections=detections)
                inside  = detections[mask]
                ids_now = set(inside.tracker_id.tolist()) if inside.tracker_id is not None else set()
                zone.total_in += len(ids_now - zone.track_ids_inside)
                zone.track_ids_inside = ids_now
                zone.count = len(ids_now)
                type_counts = defaultdict(int)
                if inside.class_id is not None:
                    for cid in inside.class_id:
                        type_counts[VEHICLE_CLASSES.get(int(cid), "vehicle")] += 1
                zone_stats[zone.name] = {
                    "current": zone.count,
                    "total_in": zone.total_in,
                    "types": dict(type_counts),
                }

            # Parking
            pk_zone = next((z for z in self.zones if z.zone_type=="parking"), None)
            for slot, sv_pz in zip(self.parking_slots, sv_pk_zones):
                slot.occupied = bool(sv_pz.trigger(detections=detections).any())
            if pk_zone:
                pk_zone.occupied = sum(1 for s in self.parking_slots if s.occupied)

            # Alerts
            now = time.time()
            for zone in self.zones:
                if zone.zone_type == "parking" and zone.total_slots:
                    if zone.occupancy_pct >= ALERT_PCT and (now - zone.last_alert_time) > 300:
                        zone.last_alert_time = now
                        send_alert(zone.name, f"Parking {zone.occupancy_pct:.0f}% full ({zone.occupied}/{zone.total_slots})")

            # DB write throttled
            if now - self._last_db_write > 30:
                self._last_db_write = now
                self._write_db(zone_stats)

            # Annotate
            annotated = frame.copy()
            for slot in self.parking_slots:
                color = (0, 60, 220) if slot.occupied else (0, 200, 80)
                cv2.polylines(annotated, [slot.polygon], True, color, 2)
                cx = int(slot.polygon[:, 0].mean())
                cy = int(slot.polygon[:, 1].mean())
                cv2.putText(annotated, "OCC" if slot.occupied else "FREE",
                            (cx-14, cy+5), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)

            for z_ann in zone_anns:
                annotated = z_ann.annotate(annotated)

            if len(detections):
                labels = []
                for i in range(len(detections)):
                    tid   = int(detections.tracker_id[i]) if detections.tracker_id is not None else 0
                    cid   = int(detections.class_id[i])
                    conf  = float(detections.confidence[i])
                    vtype = VEHICLE_CLASSES.get(cid, "vehicle")
                    spd   = speed_map.get(tid, 0)
                    labels.append(f"#{tid} {vtype} {conf:.0%} ~{spd}km/h")
                annotated = trace_ann.annotate(annotated, detections=detections)
                annotated = box_ann.annotate(annotated, detections=detections)
                annotated = label_ann.annotate(annotated, detections=detections, labels=labels)

            fps = 1.0 / max(time.time() - t0, 1e-6)
            self._fps_history.append(fps)
            avg_fps = sum(self._fps_history) / len(self._fps_history)
            cv2.putText(annotated, f"FPS: {avg_fps:.1f}", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
            cv2.putText(annotated, datetime.now().strftime("%Y-%m-%d  %H:%M:%S"),
                        (10, frame_h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1, cv2.LINE_AA)

            stats_payload = {
                "fps": round(avg_fps, 1),
                "zones": zone_stats,
                "parking": {
                    "slots": [{"id": s.slot_id, "occupied": s.occupied} for s in self.parking_slots],
                    "summary": {
                        z.name: {"total": z.total_slots, "occupied": z.occupied,
                                 "free": z.total_slots - z.occupied, "pct": z.occupancy_pct}
                        for z in self.zones if z.zone_type == "parking"
                    },
                },
                "timestamp": datetime.now().isoformat(),
                "error": None,
            }

            with self._lock:
                self.latest_frame = annotated.copy()
                self.stats = stats_payload

            if self.socketio:
                self.socketio.emit("stats_update", stats_payload)

        cap.release()
        print("[Engine] Stream closed.")

    def _emit_error(self, msg):
        payload = {"error": msg, "fps": 0, "zones": {}, "parking": {"slots": [], "summary": {}}, "timestamp": datetime.now().isoformat()}
        with self._lock:
            self.stats = payload
        if self.socketio:
            self.socketio.emit("stats_update", payload)

    def _write_db(self, zone_stats):
        session = Session()
        try:
            for zname, zdata in zone_stats.items():
                for vtype, cnt in zdata.get("types", {}).items():
                    session.add(VehicleCount(zone_name=zname, count=cnt, vehicle_type=vtype))
            for zone in self.zones:
                if zone.zone_type == "parking" and zone.total_slots:
                    session.add(ParkingSnapshot(
                        zone_name=zone.name, total_slots=zone.total_slots,
                        occupied=zone.occupied, occupancy_pct=zone.occupancy_pct,
                    ))
            session.commit()
        except Exception as e:
            print(f"[DB] Write error: {e}")
            session.rollback()
        finally:
            session.close()

    def get_jpeg_frame(self):
        with self._lock:
            if self.latest_frame is None:
                return None
            _, buf = cv2.imencode(".jpg", self.latest_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            return buf.tobytes()