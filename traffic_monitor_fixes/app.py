"""
Traffic & Parking Monitor — Flask + SocketIO application.
"""

import os, json
from datetime import datetime, timedelta
from io import BytesIO

from dotenv import load_dotenv
load_dotenv()

from flask import Flask, render_template, Response, jsonify, request, send_file
from flask_socketio import SocketIO

from utils.models import init_db, Session, VehicleCount, ParkingSnapshot, AlertLog
from utils.engine import DetectionEngine
from utils.alerts import send_alert

# ── App setup ────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "dev_secret_key_change_me")
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

init_db()
engine = DetectionEngine(socketio=socketio)

# ── Demo zones on a 1280×720 frame ───────────────────────────────────────────
# These work great for a standard traffic camera or demo video.
# Adjust coordinates via the /configure page for your specific camera.
engine.add_counting_zone("North Lane", [
    [100, 200], [540, 200], [540, 420], [100, 420]
])
engine.add_counting_zone("South Lane", [
    [540, 200], [980, 200], [980, 420], [540, 420]
])
engine.add_parking_zone("Parking Lot A", slots=[
    # Row 1 — 5 slots
    [[60,  460], [160, 460], [160, 560], [60,  560]],
    [[170, 460], [270, 460], [270, 560], [170, 560]],
    [[280, 460], [380, 460], [380, 560], [280, 560]],
    [[390, 460], [490, 460], [490, 560], [390, 560]],
    [[500, 460], [600, 460], [600, 560], [500, 560]],
    # Row 2 — 5 slots
    [[60,  570], [160, 570], [160, 660], [60,  660]],
    [[170, 570], [270, 570], [270, 660], [170, 660]],
    [[280, 570], [380, 570], [380, 660], [280, 660]],
    [[390, 570], [490, 570], [490, 660], [390, 660]],
    [[500, 570], [600, 570], [600, 660], [500, 660]],
])

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/start", methods=["POST"])
def api_start():
    data   = request.json or {}
    source = data.get("source", "0")
    engine.stop()
    import time as _time
    _time.sleep(0.3)  # let previous thread fully stop
    started = engine.start(source)

    if not started:
        return jsonify({
            "status": "error",
            "message": engine.last_error or "Failed to start video source."
        }), 400

    return jsonify({"status": "started", "source": source})


@app.route("/api/status")
def api_status():
    """Poll this after /api/start to check if the stream actually opened."""
    has_frame = engine.latest_frame is not None
    return jsonify({
        "running": has_frame,
        "error": engine.last_error,
    })


@app.route("/api/stop", methods=["POST"])
def api_stop():
    engine.stop()
    return jsonify({"status": "stopped"})


@app.route("/api/stats")
def api_stats():
    return jsonify(engine.stats)


@app.route("/api/history")
def api_history():
    hours  = int(request.args.get("hours", 24))
    since  = datetime.utcnow() - timedelta(hours=hours)
    session = Session()

    counts = session.query(VehicleCount).filter(VehicleCount.timestamp >= since).all()
    parking = session.query(ParkingSnapshot).filter(ParkingSnapshot.timestamp >= since).all()
    alerts  = session.query(AlertLog).filter(AlertLog.timestamp >= since).all()
    session.close()

    return jsonify({
        "vehicle_counts": [
            {"time": r.timestamp.isoformat(), "zone": r.zone_name,
             "type": r.vehicle_type, "count": r.count}
            for r in counts
        ],
        "parking_snapshots": [
            {"time": r.timestamp.isoformat(), "zone": r.zone_name,
             "total": r.total_slots, "occupied": r.occupied, "pct": r.occupancy_pct}
            for r in parking
        ],
        "alerts": [
            {"time": r.timestamp.isoformat(), "zone": r.zone_name,
             "msg": r.message, "sms": r.sent_sms}
            for r in alerts
        ],
    })


@app.route("/video_feed")
def video_feed():
    def generate():
        while True:
            frame = engine.get_jpeg_frame()
            if frame:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            else:
                import time; time.sleep(0.05)
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/report/pdf")
def api_pdf_report():
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.units import cm

    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=2*cm, rightMargin=2*cm,
                             topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("Title2", parent=styles["Title"], fontSize=18, spaceAfter=6)
    h2_style    = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=12, spaceBefore=12)

    story = []
    story.append(Paragraph("Traffic & Parking Monitor — Report", title_style))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]))
    story.append(Spacer(1, 0.4*cm))

    # Live stats
    stats = engine.stats
    if stats:
        story.append(Paragraph("Live Zone Counts", h2_style))
        rows = [["Zone", "Vehicles Now", "Total Entered"]]
        for zname, zdata in stats.get("zones", {}).items():
            rows.append([zname, str(zdata.get("current", 0)), str(zdata.get("total_in", 0))])
        t = Table(rows, colWidths=[6*cm, 4*cm, 4*cm])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a1a2e")),
            ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
            ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#f0f0f0"), colors.white]),
            ("GRID",       (0, 0), (-1, -1), 0.5, colors.grey),
            ("ALIGN",      (1, 0), (-1, -1), "CENTER"),
        ]))
        story.append(t)
        story.append(Spacer(1, 0.4*cm))

        # Parking summary
        story.append(Paragraph("Parking Summary", h2_style))
        for zname, pdata in stats.get("parking", {}).get("summary", {}).items():
            pct  = pdata.get("pct", 0)
            color_hex = "#d32f2f" if pct >= 85 else "#f57c00" if pct >= 60 else "#2e7d32"
            row_data = [
                [Paragraph(f"<b>{zname}</b>", styles["Normal"]),
                 f"{pdata['occupied']}/{pdata['total']} slots",
                 Paragraph(f'<font color="{color_hex}"><b>{pct}%</b></font>', styles["Normal"])]
            ]
            t2 = Table(row_data, colWidths=[6*cm, 4*cm, 4*cm])
            t2.setStyle(TableStyle([
                ("BOX", (0, 0), (-1, -1), 0.5, colors.grey),
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fafafa")),
            ]))
            story.append(t2)
            story.append(Spacer(1, 0.2*cm))

    # Recent alerts
    session = Session()
    recent_alerts = (session.query(AlertLog)
                     .filter(AlertLog.timestamp >= datetime.utcnow() - timedelta(hours=24))
                     .order_by(AlertLog.timestamp.desc()).limit(20).all())
    session.close()

    if recent_alerts:
        story.append(Paragraph("Recent Alerts (last 24 h)", h2_style))
        rows = [["Time", "Zone", "Message", "SMS"]]
        for a in recent_alerts:
            rows.append([
                a.timestamp.strftime("%H:%M:%S"),
                a.zone_name, a.message[:50],
                "Yes" if a.sent_sms else "No",
            ])
        t3 = Table(rows, colWidths=[2.5*cm, 3.5*cm, 7*cm, 1.5*cm])
        t3.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a1a2e")),
            ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
            ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#fff3e0"), colors.white]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
        ]))
        story.append(t3)

    doc.build(story)
    buf.seek(0)
    return send_file(buf, mimetype="application/pdf",
                     as_attachment=True,
                     download_name=f"traffic_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf")


@app.route("/api/test_alert", methods=["POST"])
def test_alert():
    send_alert("Test Zone", "Manual test alert from dashboard.")
    return jsonify({"status": "alert sent"})


# ── SocketIO events ───────────────────────────────────────────────────────────

@socketio.on("connect")
def on_connect():
    socketio.emit("stats_update", engine.stats)


if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, debug=False)