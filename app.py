import sys
import os
import json
import threading  # To run background tasks for AI detection
import requests  # To send notification to the user's other devices
# Force Python to recognize 'backend/' as a package
from flask_mail import Mail, Message  # ✅ Add Flask-Mail

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

# ✅ Configure Flask-Mail
app.config["MAIL_SERVER"] = "smtp.gmail.com"
app.config["MAIL_PORT"] = 587
app.config["MAIL_USE_TLS"] = True
app.config["MAIL_USERNAME"] = os.getenv("EMAIL_USER")  # ✅ Uses .env file
app.config["MAIL_PASSWORD"] = os.getenv("EMAIL_PASSWORD")  # ✅ Uses .env file
app.config["MAIL_DEFAULT_SENDER"] = os.getenv("EMAIL_USER")  # ✅ Uses your email as sender

mail = Mail(app)  # ✅ Initialize Flask-Mail


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
        return jsonify({"error": "User ID is required"}), 400

    print(f"🔄 Received Bluetooth disconnect alert for user {user_id}")

    # Fetch phone status from the database
    phone_status = PhoneStatus.query.filter_by(user_id=user_id).first()

    if phone_status:
        # ✅ Ensure `last_location` is not None
        location_name = phone_status.last_location if phone_status.last_location else "Unknown Location"

        print(f"🚨 User {user_id} may have left their phone at {location_name}!")
        send_alert(user_id, location_name)

        return jsonify({"message": "Disconnection alert received & notification sent"}), 200

    print(f"⚠️ No phone status found for user {user_id}")
    return jsonify({"message": "Disconnection alert received, but no stored location"}), 200


def send_alert(user_id, location_name, recipient_email):
    """Send an email alert when the user leaves their phone behind."""
    try:
        msg = Message(
            subject="📍 Phone Left Behind Alert!",
            recipients=[recipient_email],  # ✅ Replace with user's email
            body=f"Hey! It looks like you left your phone at {location_name}. Please check!"
        )
        mail.send(msg)
        print(f"✅ Email alert sent to user {recipient_email} about {location_name}")

    except Exception as e:
        print(f"❌ Error sending email: {e}")




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

@app.route("/debug/routes", methods=["GET"])
def list_routes():
    """Return a list of all available routes in the Flask app."""
    routes = []
    for rule in app.url_map.iter_rules():
        routes.append({
            "endpoint": rule.endpoint,
            "methods": list(rule.methods),
            "route": str(rule)
        })
    
    return jsonify({"routes": routes})

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Phonelert API is Running!"}), 200


@app.route("/check-location", methods=["POST"])
def check_location():
    """Receive location data from React Native and send an email alert if the phone is left behind."""
    data = request.json
    location_name = data.get("locationName")
    recipient_email = data.get("email")  # Optional: Send user ID if available

    if not location_name or not recipient_email:
        return jsonify({"error": "Missing location name or email"}), 400

    print(f"📍 Phone left at {location_name}. Sending email to {recipient_email}... ")

    send_alert(user_id, location_name, recipient_email)  # 🔹 Call the function to send an email

    return jsonify({"message": f"✅ Email sent to {recipient_email} about {location_name}."}), 200



@app.route("/test-email", methods=["GET"])
def test_email():
    """Send a test email to verify the setup"""
    try:
        msg = Message(
            "🔔 Phonelert Test Email",
            recipients=["adhithyahere7@gmail.com"],  # ✅ Change to your email
            body="Hello! This is a test email from Phonelert to verify email alerts.",
        )
        mail.send(msg)
        return jsonify({"message": "✅ Test email sent successfully!"}), 200
    except Exception as e:
        return jsonify({"error": f"❌ Failed to send email: {str(e)}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
