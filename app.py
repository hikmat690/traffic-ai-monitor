import os
from datetime import datetime, timedelta
from io import BytesIO

from dotenv import load_dotenv
load_dotenv()

from flask import Flask, render_template, Response, jsonify, request, send_file
from flask_socketio import SocketIO

from utils.models import init_db, Session, VehicleCount, ParkingSnapshot, AlertLog
from utils.engine import DetectionEngine
from utils.alerts import send_alert

app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "dev_secret_key")
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

init_db()
engine = DetectionEngine(socketio=socketio)

engine.add_counting_zone("North Lane", [[100,200],[540,200],[540,420],[100,420]])
engine.add_counting_zone("South Lane", [[540,200],[980,200],[980,420],[540,420]])
engine.add_parking_zone("Parking Lot A", slots=[
    [[60,460],[160,460],[160,560],[60,560]],
    [[170,460],[270,460],[270,560],[170,560]],
    [[280,460],[380,460],[380,560],[280,560]],
    [[390,460],[490,460],[490,560],[390,560]],
    [[500,460],[600,460],[600,560],[500,560]],
    [[60,570],[160,570],[160,660],[60,660]],
    [[170,570],[270,570],[270,660],[170,660]],
    [[280,570],[380,570],[380,660],[280,660]],
    [[390,570],[490,570],[490,660],[390,660]],
    [[500,570],[600,570],[600,660],[500,660]],
])

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/start", methods=["POST"])
def api_start():
    data = request.json or {}
    source = data.get("source", "0")
    engine.stop()
    import time; time.sleep(0.5)
    engine.start(source)
    return jsonify({"status": "started", "source": source})

@app.route("/api/stop", methods=["POST"])
def api_stop():
    engine.stop()
    return jsonify({"status": "stopped"})

@app.route("/api/stats")
def api_stats():
    return jsonify(engine.stats)

@app.route("/api/history")
def api_history():
    hours = int(request.args.get("hours", 24))
    since = datetime.utcnow() - timedelta(hours=hours)
    session = Session()
    counts  = session.query(VehicleCount).filter(VehicleCount.timestamp >= since).all()
    parking = session.query(ParkingSnapshot).filter(ParkingSnapshot.timestamp >= since).all()
    alerts  = session.query(AlertLog).filter(AlertLog.timestamp >= since).all()
    session.close()
    return jsonify({
        "vehicle_counts": [{"time": r.timestamp.isoformat(), "zone": r.zone_name, "type": r.vehicle_type, "count": r.count} for r in counts],
        "parking_snapshots": [{"time": r.timestamp.isoformat(), "zone": r.zone_name, "total": r.total_slots, "occupied": r.occupied, "pct": r.occupancy_pct} for r in parking],
        "alerts": [{"time": r.timestamp.isoformat(), "zone": r.zone_name, "msg": r.message, "sms": r.sent_sms} for r in alerts],
    })

@app.route("/video_feed")
def video_feed():
    def generate():
        import time
        while True:
            frame = engine.get_jpeg_frame()
            if frame:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            else:
                time.sleep(0.05)
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/report/pdf")
def api_pdf_report():
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.units import cm
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=2*cm, rightMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    story = [Paragraph("Traffic & Parking Monitor Report", styles["Title"]),
             Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]),
             Spacer(1, 0.5*cm)]
    stats = engine.stats
    if stats:
        rows = [["Zone", "Vehicles Now", "Total Entered"]]
        for zname, zdata in stats.get("zones", {}).items():
            rows.append([zname, str(zdata.get("current", 0)), str(zdata.get("total_in", 0))])
        t = Table(rows, colWidths=[6*cm, 4*cm, 4*cm])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1a1a2e")),
            ("TEXTCOLOR",  (0,0), (-1,0), colors.white),
            ("GRID",       (0,0), (-1,-1), 0.5, colors.grey),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.HexColor("#f0f0f0"), colors.white]),
        ]))
        story.append(t)
    doc.build(story)
    buf.seek(0)
    return send_file(buf, mimetype="application/pdf", as_attachment=True,
                     download_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf")

@app.route("/api/test_alert", methods=["POST"])
def test_alert():
    send_alert("Test Zone", "Manual test alert.")
    return jsonify({"status": "alert sent"})

@socketio.on("connect")
def on_connect():
    socketio.emit("stats_update", engine.stats)

if __name__ == "__main__":
    print("=" * 50)
    print(" TrafficAI starting on http://localhost:5000")
    print("=" * 50)
    socketio.run(app, host="0.0.0.0", port=5000, debug=False, allow_unsafe_werkzeug=True)