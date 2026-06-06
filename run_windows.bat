@echo off
echo ============================================
echo  TrafficAI — Starting server...
echo ============================================
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)
call venv\Scripts\activate
pip install -r requirements.txt -q
python app.py
pause
