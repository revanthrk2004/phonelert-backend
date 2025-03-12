import time
from datetime import datetime
from geopy.distance import geodesic
import requests
from database.models import PhoneStatus, UserLocation
from flask import current_app
from database.db_manager import db

# Define inactivity duration (e.g., 5 minutes = 300 seconds)
INACTIVITY_THRESHOLD = 300  

def check_forgotten_phone(user_id, user_lat, user_lon, is_moving):
    """AI-based Detection: Check if user forgot their phone"""
    
    with current_app.app_context():  # Ensure Flask context is available
        # Fetch user's registered locations
        locations = UserLocation.query.filter_by(user_id=user_id).all()
        if not locations:
            return

        for loc in locations:
            loc_coords = (loc.latitude, loc.longitude)
            user_coords = (user_lat, user_lon)

            distance = geodesic(loc_coords, user_coords).meters  # Calculate distance

            if distance <= loc.radius:  # User is near a registered location
                print(f"📍 User is near {loc.name} (Distance: {distance:.2f}m)")
                
                # Fetch phone status
                phone_status = PhoneStatus.query.filter_by(user_id=user_id).first()

                if phone_status is None:
                    phone_status = PhoneStatus(user_id=user_id, last_location=loc.name, last_motion_time=datetime.utcnow())
                    db.session.add(phone_status)
                    db.session.commit()

                # Check if the phone is unused (not moving) for 5+ minutes
                if not is_moving:
                    if phone_status.last_motion_time is None:
                        phone_status.last_motion_time = datetime.utcnow()
                    else:
                        elapsed_time = (datetime.utcnow() - phone_status.last_motion_time).total_seconds()
                        if elapsed_time >= INACTIVITY_THRESHOLD:
                            print("🚨 ALERT: Phone may have been forgotten at", loc.name)
                            send_alert(user_id, loc.name)

                    db.session.commit()
                else:
                    # Update motion time if phone is moving
                    phone_status.last_motion_time = datetime.utcnow()
                    db.session.commit()

def send_alert(user_id, location_name):
    """Send alert to user's secondary device when phone is left behind"""

    notification_data = {
        "title": "🚨 Forgot Your Phone?",
        "message": f"You left your phone at {location_name}!",
        "user_id": user_id
    }

    # Send alert to the notification service
    notification_url = "https://phonelert-backend.onrender.com/notify"
    response = requests.post(notification_url, json=notification_data)

    if response.status_code == 200:
        print(f"✅ Notification sent: {notification_data}")
    else:
        print(f"❌ Failed to send notification: {response.text}")
