import sys
import os

import threading  # To run background tasks for AI detection
import requests  # To send notification to the user's other devices
# Force Python to recognize 'backend/' as a package

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from flask import Flask, request, jsonify
from flask_cors import CORS
from database.db_manager import create_app, db
from routes.auth_route import auth
  
from flask_sqlalchemy import SQLAlchemy  # Database to store locations
from geopy.distance import geodesic  # To calculate distance between two coordinates
from datetime import datetime  # For timestamping last phone activity
from database.models import PhoneStatus  # ✅ Import the model
from flask_migrate import Migrate
from sqlalchemy import inspect

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



@app.route("/bluetooth/disconnect", methods=["POST"])
def bluetooth_disconnect():
    """Triggered when smartwatch disconnects from phone."""
    data = request.json
    user_id = data.get("user_id")

    if not user_id:
        print("❌ Missing user ID in request!")
        return jsonify({"error": "User ID is required"}), 400

    print(f"🔄 Received Bluetooth disconnect alert for user {user_id}")

    # Fetch user's registered safe locations
    phone_status = PhoneStatus.query.filter_by(user_id=user_id).first()

    if phone_status:
        # 🚨 ALERT: Smartwatch Disconnected, Check If Phone Is Left Behind
        print(f"🚨 User {user_id} may have left their phone at {phone_status.last_location}!")
        send_alert(user_id, phone_status.last_location)
        return jsonify({"message": "Disconnection alert received & notification sent"}), 200

    print(f"⚠️ No phone status found for user {user_id}")
    return jsonify({"message": "Disconnection alert received, but no stored location"}), 200



def send_alert(user_id, location_name):
    """Send notification when phone is left behind."""
    notification_data = {
        "title": "🚨 Forgot Your Phone?",
        "message": f"You left your phone at {location_name}!",
        "user_id": user_id
    }
    notification_url = "https://phonelert-backend.onrender.com/notify"
    response = requests.post(notification_url, json=notification_data)

    if response.status_code == 200:
        print(f"✅ Notification sent: {notification_data}")
    else:
        print(f"❌ Failed to send notification: {response.text}")


def ai_background_task():
    """Background AI Task to Detect Forgotten Phones"""
    with app.app_context():  # ✅ Use app instead of current_app
        while True:
            try:
                inspector = inspect(db.engine)
                if "phone_status" in inspector.get_table_names():
                    users = PhoneStatus.query.all()
                    for user in users:
                        last_location = (user.last_known_latitude, user.last_known_longitude)
                        registered_location = (user.registered_latitude, user.registered_longitude)

                        distance = geodesic(last_location, registered_location).meters

                        if distance < 50 and not user.is_moving and (time.time() - user.last_motion_time.timestamp()) > 300:
                            print(f"User {user.user_id} might have left their phone!")
                            send_alert(user.user_id, user.registered_location)

                else:
                    print("PhoneStatus table does not exist yet.")

            except Exception as e:
                print(f"Error in AI monitoring task: {e}")

            time.sleep(60)  # Run check every 60 seconds

if __name__ == "__main__":
    app.run(debug=True)
