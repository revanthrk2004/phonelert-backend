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

tracking_users = {}  # ✅ Store tracking status per user

def send_email_alert(user_id):
    """Sends an alert email with a stop-tracking link and Google Maps location."""
    with app.app_context():
        phone_status = PhoneStatus.query.filter_by(user_id=user_id).first()

        if not phone_status:
            print(f"⚠️ No phone status found for user {user_id}.")
            return

        recipient_emails = tracking_users.get(user_id, {}).get("emails", [])

        last_lat, last_long = phone_status.last_latitude, phone_status.last_longitude
        google_maps_link = f"https://www.google.com/maps?q={last_lat},{last_long}"
        stop_tracking_link = f"https://phonelert-backend.onrender.com/stop-tracking?user_id={user_id}"

        subject = "🚨 Urgent: Your Phone is Still Left Behind!"
        body = f"""
        Your phone has not been retrieved yet. Please check its last known location!
        
        📍 **Last Known Location:** {google_maps_link}

        🛑 **Stop Tracking:** Click here to stop alerts → [Stop Tracking]({stop_tracking_link})
        """

        for email in recipient_emails:
            try:
                msg = Message(subject, recipients=[email], body=body)
                mail.send(msg)
                print(f"✅ Email sent to {email}")
            except Exception as e:
                print(f"❌ Failed to send email to {email}: {str(e)}")



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


@app.route("/start-tracking", methods=["POST"])
def start_tracking():
    """Activates live tracking only if the phone stays in one place for 3 minutes."""
    data = request.json
    user_id = data.get("user_id")
    recipient_emails = data.get("emails", [])

    if not user_id or not recipient_emails:
        return jsonify({"error": "User ID and emails are required"}), 400

    if user_id in tracking_users and tracking_users[user_id]["active"]:
        return jsonify({"message": "Tracking is already active for this user"}), 200

    tracking_users[user_id] = {"active": True, "emails": recipient_emails}

    tracking_thread = threading.Thread(target=send_repeated_alerts, args=(user_id, recipient_emails), daemon=True)
    tracking_thread.start()

    print(f"🚀 Tracking started for user {user_id}, checking if phone stays in one place for 3 minutes.")

    return jsonify({"message": "✅ Tracking started. If phone stays in one place for 3 minutes, an alert will be sent."}), 200





def send_repeated_alerts(user_id, recipient_emails):
    """Sends email alerts only if the phone remains in the same location for 3 minutes."""
    with app.app_context():
        phone_status = PhoneStatus.query.filter_by(user_id=user_id).first()

        if not phone_status:
            print(f"⚠️ No phone status found for user {user_id}.")
            return

        # ✅ Initialize last known location
        last_lat, last_long = phone_status.last_latitude, phone_status.last_longitude
        last_update_time = datetime.utcnow()

        while tracking_users.get(user_id, {}).get("active", False):  # ✅ Corrected indentation
            time.sleep(180)  # ✅ Wait for 3 minutes
            
            # ✅ Fetch latest phone status from database
            phone_status = PhoneStatus.query.filter_by(user_id=user_id).first()
            if not phone_status:
                print(f"⚠️ No phone status found for user {user_id}. Stopping tracking.")
                break

            # ✅ Retrieve current latitude and longitude
            current_lat, current_long = phone_status.last_latitude, phone_status.last_longitude

            # ✅ If phone hasn't moved, check if it's been 3 minutes
            if (current_lat, current_long) == (last_lat, last_long):
                time_elapsed = (datetime.utcnow() - last_update_time).total_seconds()

                if time_elapsed >= 180:  # ✅ If 3 minutes have passed
                    print(f"📌 Phone has stayed in the same location for 3 minutes. Sending alert...")
                    send_email_alert(user_id)  # ✅ Send email alert
            else:
                # ✅ If phone moved, reset timer
                last_lat, last_long = current_lat, current_long
                last_update_time = datetime.utcnow()











@app.route("/stop-tracking", methods=["GET", "POST"])
def stop_tracking():
    """Stops the repeated email alerts when user clicks the stop link."""
    user_id = request.args.get("user_id") or request.json.get("user_id")

    if not user_id or user_id not in tracking_users:
        return jsonify({"error": "Tracking was not active for this user"}), 400

    tracking_users[user_id]["active"] = False
    print(f"🛑 Stopped tracking for user {user_id}")

    return jsonify({"message": "✅ Tracking stopped successfully"}), 200





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
