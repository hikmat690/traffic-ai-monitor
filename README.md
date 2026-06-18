# TrafficAI — Live Traffic \& Parking Monitor

### Advanced AI-powered real-time traffic analysis system

!\[Python](https://img.shields.io/badge/Python-3.10+-blue)
!\[YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple)
!\[Flask](https://img.shields.io/badge/Flask-3.0-green)
!\[License](https://img.shields.io/badge/License-MIT-yellow)

\---

## Features

* **Real-time vehicle detection** — cars, trucks, motorcycles, buses (YOLOv8)
* **Multi-object tracking** — unique ID per vehicle (ByteTrack)
* **Speed estimation** — approximate km/h per tracked vehicle
* **Counting zones** — draw virtual lines/polygons, count entries
* **Parking lot monitor** — per-slot occupancy, green/red grid
* **Live dashboard** — real-time charts, FPS display, timestamp
* **SMS alerts** — Twilio integration when lot exceeds threshold
* **PDF reports** — one-click professional report download
* **SQLite history** — all counts \& snapshots stored automatically
* **YouTube/RTSP support** — test with any live camera stream

\---



!\[TrafficAI Dashboard](demo.png)

## Quick Start (5 minutes)

### 1\. Clone / download the project

```bash
git clone https://github.com/yourname/traffic-monitor.git
cd traffic-monitor
```

### 2\. Create a virtual environment

```bash
# Windows
python -m venv venv
venv\\Scripts\\activate

# Mac / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3\. Install dependencies

```bash
pip install -r requirements.txt
```

> First run downloads YOLOv8n weights (\~6 MB) automatically.

### 4\. Configure environment (optional)

```bash
cp .env.example .env
# Edit .env with your Twilio keys if you want SMS alerts
# Leave as-is to run without SMS
```

### 5\. Run the app

```bash
python app.py
```

### 6\. Open browser

```
http://localhost:5000
```

\---

## Usage

### Source options (enter in the dashboard input box)

|Input|Meaning|
|-|-|
|`0`|Default webcam|
|`1`|Second camera|
|`video.mp4`|Local video file|
|`rtsp://user:pass@192.168.1.10/stream`|IP camera RTSP stream|
|`https://www.youtube.com/watch?v=xxxxxx`|YouTube live stream|

### Recommended demo video (free)

Search YouTube for **"traffic camera live"** — copy any URL and paste it into the source box.

\---

## SMS Alerts (optional — free tier)

1. Sign up at https://www.twilio.com (free trial gives \~$15 credit)
2. Copy your `Account SID`, `Auth Token`, and a Twilio phone number
3. Paste into your `.env` file
4. Alerts fire automatically when parking lot exceeds 85% occupancy

\---

## Project Structure

```
traffic\_monitor/
├── app.py                  # Flask app + all API routes
├── requirements.txt        # All dependencies
├── .env.example            # Config template
├── utils/
│   ├── engine.py           # YOLOv8 + ByteTrack + zone logic
│   ├── models.py           # SQLite database models
│   └── alerts.py           # SMS alert sender
├── templates/
│   └── index.html          # Full dashboard UI
└── data/
    └── traffic.db          # Auto-created SQLite database
```

\---

## Customizing Zones

Edit the zone definitions in `app.py` (lines 30–55).
Each zone is a list of `\[x, y]` pixel coordinates forming a polygon.
Use a tool like https://roboflow.com/annotate to get coordinates from your camera frame.

\---

## Tech Stack

|Tool|Purpose|
|-|-|
|YOLOv8n|Vehicle detection|
|supervision|ByteTrack + annotators|
|Flask + SocketIO|Web server + real-time push|
|Chart.js|Live charts in browser|
|SQLite + SQLAlchemy|Data persistence|
|ReportLab|PDF report generation|
|Twilio|SMS alerts|
|yt-dlp|YouTube stream extraction|

\---

## License

MIT — free to use, modify, and deploy.

