import requests
from database.models import User

def send_alert_to_secondary_device(user_id, latitude, longitude):
    """Send alert to secondary device when phone is forgotten."""
    try:
        user = User.query.filter_by(id=user_id).first()
        if user and user.secondary_device_token:
            notification_data = {
                "title": "Forgotten Phone Alert! 🚨",
                "body": f"Your phone might be left at {latitude}, {longitude}",
                "token": user.secondary_device_token,
            }
            send_notification(notification_data)
            print("🚨 Alert sent to secondary device!")
    except Exception as e:
        print("❌ Error sending alert:", e)

def send_notification(data):
    """Simulate sending push notifications."""
    try:
        notification_url = "https://phonelert-backend.onrender.com/notify"
        response = requests.post(notification_url, json=data)
        if response.status_code == 200:
            print("✅ Notification sent:", data)
        else:
            print("❌ Notification failed:", response.text)
    except Exception as e:
        print("❌ Error sending notification:", e)
