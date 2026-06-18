# TrafficAI - Live Traffic \& Parking Monitor

### Advanced AI-powered real-time traffic analysis system

!\[Python](https://img.shields.io/badge/Python-3.10+-blue)
!\[YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple)
!\[Flask](https://img.shields.io/badge/Flask-3.0-green)
!\[License](https://img.shields.io/badge/License-MIT-yellow)

\---

!\[TrafficAI Dashboard](demo.png)

\---

## Features

* **Real-time vehicle detection** - cars, trucks, motorcycles, buses (YOLOv8)
* **Multi-object tracking** - unique ID per vehicle (ByteTrack)
* **Speed estimation** - approximate km/h per tracked vehicle
* **Counting zones** - draw virtual lines/polygons, count entries
* **Parking lot monitor** - per-slot occupancy, green/red grid
* **Live dashboard** - real-time charts, FPS display, timestamp
* **SMS alerts** - Twilio integration when lot exceeds threshold
* **PDF reports** - one-click professional report download
* **SQLite history** - all counts and snapshots stored automatically
* **YouTube/RTSP support** - test with any live camera stream

\---

## Quick Start

### 1\. Clone the project

```bash
git clone https://github.com/hikmat690/traffic-ai-monitor.git
cd traffic-ai-monitor
```

### 2\. Create virtual environment

```bash
python -m venv venv
venv\\\\Scripts\\\\activate
```

### 3\. Install dependencies

```bash
pip install -r requirements.txt
```

### 4\. Run the app

```bash
python app.py
```

### 5\. Open browser

```
http://localhost:5000
```

\---

## Usage

|Input|Meaning|
|-|-|
|`0`|Default webcam|
|`1`|Second camera|
|`video.mp4`|Local video file|
|`rtsp://ip/stream`|IP camera|
|YouTube URL|YouTube live stream|

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

## SMS Alerts (optional)

1. Sign up at https://www.twilio.com (free trial)
2. Add your keys to `.env` file
3. Alerts fire automatically when parking lot exceeds 85%

\---

## License

MIT - free to use, modify, and deploy.



