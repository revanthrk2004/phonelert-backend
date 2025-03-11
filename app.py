import sys
import os
import time  # To track inactivity duration
import threading  # To run background tasks for AI detection
import requests  # To send notification to the user's other devices
# Force Python to recognize 'backend/' as a package
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from flask import Flask, request, jsonify, current_app
from flask_cors import CORS
from database.db_manager import create_app, db
from routes.auth_route import auth
from routes.location_route import location_bp  
from flask_sqlalchemy import SQLAlchemy  # Database to store locations
from geopy.distance import geodesic  # To calculate distance between two coordinates
from datetime import datetime  # For timestamping last phone activity
from database.models import PhoneStatus  # ✅ Import the model
from flask_migrate import Migrate

app = create_app()
migrate = Migrate(app, db)  # ✅ Enable migrations


# ✅ Enable CORS for all requests
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

@app.route('/auth/login', methods=['POST'])
def login():
    data = request.json
    email = data.get('email')
    password = data.get('password')

    if not email or not password:
        return jsonify({"error": "Missing email or password"}), 400

    if email == "revanthkkrishnan@gmail.com" and password == "Kinnu2004@@@":
        return jsonify({"message": "Login successful"}), 200
    else:
        return jsonify({"error": "Invalid credentials"}), 401

@app.before_request
def handle_options_request():
    """Handle CORS preflight requests"""
    if request.method == "OPTIONS":
        response = jsonify({"message": "CORS preflight request success"})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Methods", "GET, POST, OPTIONS, DELETE, PUT")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type, Authorization")
        return response, 200

def check_forgotten_phone(user_id, user_lat, user_lon, is_moving):
    """AI Detection to check if user forgot their phone"""

    # Fetch user's registered locations
    locations = UserLocation.query.filter_by(user_id=user_id).all()
    if not locations:
        return

    for loc in locations:
        loc_coords = (loc.latitude, loc.longitude)
        user_coords = (user_lat, user_lon)

        distance = geodesic(loc_coords, user_coords).meters  # Calculate distance in meters

        if distance <= loc.radius:  # User is near the registered location
            print(f"📍 User is near {loc.name} (Distance: {distance:.2f}m)")
            
            # Fetch phone status
            phone_status = PhoneStatus.query.filter_by(user_id=user_id).first()

            if phone_status is None:
                phone_status = PhoneStatus(user_id=user_id, last_location=loc.name)
                db.session.add(phone_status)
                db.session.commit()

            # Check if the phone is stationary for 5 minutes
            if not is_moving:
                if phone_status.last_motion_time is None:
                    phone_status.last_motion_time = datetime.utcnow()
                else:
                    elapsed_time = (datetime.utcnow() - phone_status.last_motion_time).total_seconds()
                    if elapsed_time >= 300:  # 5 minutes
                        print("🚨 ALERT: Phone may have been forgotten at", loc.name)
                        send_alert(user_id, loc.name)

                db.session.commit()
            else:
                # Update motion time since the phone is moving
                phone_status.last_motion_time = datetime.utcnow()
                db.session.commit()


def send_alert(user_id, location_name):
    """Send alert to user's secondary device when phone is left behind"""

    # Simulating a notification to a secondary device (smartwatch, another phone)
    notification_data = {
        "title": "🚨 Forgot Your Phone?",
        "message": f"You left your phone at {location_name}!",
        "user_id": user_id
    }

    # Assume another endpoint `/notify` handles notifications
    notification_url = "https://phonelert-backend.onrender.com/notify"
    response = requests.post(notification_url, json=notification_data)

    if response.status_code == 200:
        print(f"✅ Notification sent: {notification_data}")
    else:
        print(f"❌ Failed to send notification: {response.text}")


def ai_background_task():
    with current_app.app_context():
        users = PhoneStatus.query.all()  # Ensure database session is available
        for user in users:
            print(f"Checking status for user {user.id}")  
            time.sleep(60)  # Check every 60 seconds

# ✅ Start AI background task in a separate thread
threading.Thread(target=ai_background_task, daemon=True).start()

@app.route("/ai/detect", methods=["POST"])
def ai_detect():
    """Receive data from frontend and trigger AI detection"""
    data = request.json
    user_id = data.get("user_id")
    latitude = data.get("latitude")
    longitude = data.get("longitude")
    is_moving = data.get("is_moving")

    if not all([user_id, latitude, longitude, is_moving]):
        return jsonify({"error": "Missing data"}), 400

    check_forgotten_phone(user_id, latitude, longitude, is_moving)

    return jsonify({"message": "AI detection processed successfully!"}), 200
def send_alert_to_secondary_device(user_id, latitude, longitude):
    """Send alert to secondary device when phone is forgotten."""
    try:
        # Fetch user’s secondary device details from the database
        user = User.query.filter_by(id=user_id).first()
        if user and user.secondary_device_token:
            notification_data = {
                "title": "Forgotten Phone Alert!",
                "body": f"Your phone might be left at {latitude}, {longitude}",
                "token": user.secondary_device_token,
            }
            send_notification(notification_data)
            print("🚨 Alert sent to secondary device!")
    except Exception as e:
        print("❌ Error sending alert:", e)


if __name__ == "__main__":
    app.run(debug=True)
