#!/bin/bash
echo "============================================"
echo " TrafficAI — Starting server..."
echo "============================================"
if [ ! -d "venv" ]; then
  echo "Creating virtual environment..."
  python3 -m venv venv
fi
source venv/bin/activate
pip install -r requirements.txt -q
python app.py
