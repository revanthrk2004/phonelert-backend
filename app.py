import sys
import os
import json
import time
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


@app.route("/send-alert", methods=["POST"])
def send_alert():
    data = request.json
    location_name = data.get("locationName")
    emails = data.get("emails", [])

    if not emails:
        return jsonify({"error": "No emails provided"}), 400
    subject="📍 Phone Left Behind Alert!",
    body=f"Hey! It looks like you left your phone at {location_name}. Please check!"
        
    for email in emails:
        try:
            msg = Message(subject, recipients=[email], body=body)
            mail.send(msg)
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return jsonify({"success": True, "message": "Emails sent successfully!"})
tracking_users = {}  # ✅ Store tracking status per user

@app.route("/start-tracking", methods=["POST"])
def start_tracking():
    """Activates live tracking for a user."""
    data = request.json
    user_id = data.get("user_id")
    recipient_emails = data.get("emails", [])

    if not user_id or not recipient_emails:
        return jsonify({"error": "User ID and emails are required"}), 400

    # ✅ Stop existing tracking if it's running
    if tracking_users.get(user_id, False):
        tracking_users[user_id] = False
        time.sleep(1)  # ⏳ Small delay to allow old thread to stop

    # ✅ Start new tracking
    tracking_users[user_id] = True
    tracking_thread = threading.Thread(target=send_repeated_alerts, args=(user_id, recipient_emails), daemon=True)
    tracking_thread.start()

    print(f"🚀 Tracking restarted for user {user_id}, sending alerts to {recipient_emails}")

    return jsonify({"message": "✅ Tracking started, alerts will be sent every 3 minutes"}), 200



def send_repeated_alerts(user_id, recipient_emails):
    """Continuously sends email alerts every 3 minutes until tracking is stopped."""
    with app.app_context():  # ✅ FIX: Ensure Flask app context
        while tracking_users.get(user_id, False):  # ✅ Check if tracking is active
            subject = "🚨 Urgent: Your Phone is Still Left Behind!"
            body = "Your phone has not been retrieved yet. Please check its last known location!"

            for email in recipient_emails:
                try:
                    msg = Message(subject, recipients=[email], body=body)
                    mail.send(msg)  # ✅ FIXED: Now inside app context!
                    print(f"✅ Email sent to {email}")
                except Exception as e:
                    print(f"❌ Failed to send email to {email}: {str(e)}")

            time.sleep(30)  # 🔄 Send email every 3 minutes

        print(f"🛑 Tracking stopped for user {user_id}")






@app.route("/stop-tracking", methods=["POST"])
def stop_tracking():
    """Stops the repeated email alerts."""
    data = request.json
    user_id = data.get("user_id")

    if not user_id or user_id not in tracking_users:
        return jsonify({"error": "Tracking was not active for this user"}), 400

    tracking_users[user_id] = False
    print(f"🛑 Stopped tracking for user {user_id}")

    return jsonify({"message": "Tracking stopped successfully"}), 200



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
    recipient_emails = data.get("emails", [])  # Optional: Send user ID if available

    if not location_name or not recipient_emails:
        return jsonify({"error": "Missing location name or emails"}), 400

    if not isinstance(recipient_emails, list):  # ✅ Convert single email to list
        recipient_emails = [recipient_emails]

    print(f"📍 Phone left at {location_name}. Sending emails to {recipient_emails}...")


    subject = f"🚨 Alert: Possible Phone Left at {location_name}"
    body = f"Your phone might have been left at {location_name}. Please check your location immediately!"

    failed_emails = []
    for email in recipient_emails:
        try:
            msg = Message(subject, recipients=[email], body=body)
            mail.send(msg)
            print(f"✅ Email successfully sent to {email}")
        except Exception as e:
            failed_emails.append(email)
            print(f"❌ Failed to send email to {email}: {str(e)}")

    if failed_emails:    
            return jsonify({"error": f"Failed to send emails to: {', '.join(failed_emails)}"}), 500

    return jsonify({"message": f"✅ Emails sent to: {', '.join(recipient_emails)}"}), 200



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
