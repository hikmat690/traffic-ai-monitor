import os
from datetime import datetime
from utils.models import Session, AlertLog


def send_alert(zone_name: str, message: str):
    """Send SMS alert via Twilio and log it to the database."""
    sent = False
    sid  = os.getenv("TWILIO_ACCOUNT_SID", "")
    tok  = os.getenv("TWILIO_AUTH_TOKEN", "")
    frm  = os.getenv("TWILIO_FROM_NUMBER", "")
    to   = os.getenv("ALERT_TO_NUMBER", "")

    if sid and tok and frm and to and not sid.startswith("your_"):
        try:
            from twilio.rest import Client
            client = Client(sid, tok)
            client.messages.create(body=f"[TrafficAI] {zone_name}: {message}", from_=frm, to=to)
            sent = True
            print(f"[SMS] Sent alert → {to}")
        except Exception as e:
            print(f"[SMS] Failed: {e}")
    else:
        print(f"[ALERT] {zone_name}: {message}  (Twilio not configured — logged only)")

    session = Session()
    session.add(AlertLog(zone_name=zone_name, message=message, sent_sms=sent))
    session.commit()
    session.close()
