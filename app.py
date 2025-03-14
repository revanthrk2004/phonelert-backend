import sys
import os
import json
import time
import threading  # To run background tasks for AI detection
import requests  # To send notification to the user's other devices
# Force Python to recognize 'backend/' as a package
from flask_mail import Mail, Message  # ✅ Add Flask-Mail
from database.models import AlertHistory  # ✅ Add this line

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from flask import Flask, request, jsonify
from flask_cors import CORS
from database.db_manager import create_app, db
from routes.auth_route import auth
from database.models import PhoneStatus, UserLocation  # ✅ Import UserLocation

from flask_sqlalchemy import SQLAlchemy  # Database to store locations
from geopy.distance import geodesic  # To calculate distance between two coordinates
from datetime import datetime  # For timestamping last phone activity
from database.models import PhoneStatus  # ✅ Import the model
from flask_migrate import Migrate
from sqlalchemy import inspect
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy.exc import OperationalError


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

tracking_users = {}  # ✅ Store tracking status per user


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




@app.route("/check-location", methods=["POST"])
def check_location():
    """Receives live location data and sends an alert with the latest coordinates."""
    data = request.json
    user_id = data.get("user_id")
    latitude = data.get("latitude")  # ✅ Get real-time latitude
    longitude = data.get("longitude")  # ✅ Get real-time longitude
    recipient_emails = data.get("emails", [])  

    if not user_id or latitude is None or longitude is None or not recipient_emails:
        return jsonify({"error": "Missing required data"}), 400

    print(f"📍 Live location received: {latitude}, {longitude}")

    # ✅ Build Google Maps link using the real-time coordinates
    google_maps_link = f"https://www.google.com/maps?q={latitude},{longitude}"
    stop_tracking_link = f"https://phonelert-backend.onrender.com/stop-tracking?user_id={user_id}"

    subject = f"🚨 Urgent: Your Phone's Live Location"
    body = f"""
    Your phone's latest live location:
    
    📍 **Live Location:** {google_maps_link}

    🛑 **Stop Tracking:** Click here → [Stop Tracking]({stop_tracking_link})
    """

    failed_emails = []
    for email in recipient_emails:
        try:
            msg = Message(subject, recipients=[email], body=body)
            mail.send(msg)
            print(f"✅ Email sent to {email}")
        except Exception as e:
            failed_emails.append(email)
            print(f"❌ Failed to send email to {email}: {str(e)}")

    if failed_emails:    
        return jsonify({"error": f"Failed to send emails to: {', '.join(failed_emails)}"}), 500

    return jsonify({"message": f"✅ Live location emails sent to: {', '.join(recipient_emails)}"}), 200



def should_send_alert(user_id, live_lat, live_long):
    """
    AI Decision: Should an alert be sent?
    - If phone is in a **safe location**, don't send an alert.
    - If phone hasn't moved for **3 minutes**, send an alert.
    """
    safe_location = UserLocation.query.filter_by(user_id=user_id, location_type="safe").first()
    unsafe_location = UserLocation.query.filter_by(user_id=user_id, location_type="unsafe").first()

    # ✅ If user has saved a safe location, check distance
    if safe_location:
        distance_to_safe = geodesic((live_lat, live_long), (safe_location.latitude, safe_location.longitude)).meters
        if distance_to_safe < 50:  # ✅ If phone is within 50 meters of safe location, no alert needed
            print(f"✅ AI Decision: Phone is in SAFE location '{safe_location.name}'. No alert needed.")
            return False

    # ✅ If user has an unsafe location, check distance
    if unsafe_location:
        distance_to_unsafe = geodesic((live_lat, live_long), (unsafe_location.latitude, unsafe_location.longitude)).meters
        if distance_to_unsafe < 50:  # ✅ If phone is within unsafe location, send alert
            print(f"⚠️ AI Decision: Phone is in UNSAFE location '{unsafe_location.name}'. Sending alert!")
            return True

    # ✅ If location hasn't changed for 3 minutes, send alert
    last_alert = AlertHistory.query.filter_by(user_id=user_id).order_by(AlertHistory.timestamp.desc()).first()
    if last_alert:
        last_location = (last_alert.latitude, last_alert.longitude)
        current_location = (live_lat, live_long)
        if last_location == current_location:
            print(f"📌 AI Decision: Phone hasn't moved in 3 minutes. Sending alert!")
            return True

    # Default: Send alert if no safe location & hasn't moved
    print(f"🚨 AI Decision: No safe location found. Sending alert by default.")
    return True


def send_email_alert(user_id, live_lat=None, live_long=None):
    """Sends an alert email only if AI determines it is necessary."""
    with app.app_context():
        # 1️⃣ Try to use the live location from React Native
        if live_lat is None or live_long is None:
            phone_status = PhoneStatus.query.filter_by(user_id=user_id).first()
            if phone_status:
                live_lat, live_long = phone_status.last_latitude, phone_status.last_longitude

        # 2️⃣ If no live location, try using a saved location
        saved_location = UserLocation.query.filter_by(user_id=user_id).first()
        if (live_lat is None or live_long is None) and saved_location:
            live_lat, live_long = saved_location.latitude, saved_location.longitude
            print(f"📍 Using saved location '{saved_location.name}' instead.")

        # 3️⃣ If STILL no location, skip alert
        if live_lat is None or live_long is None:
            print(f"⚠️ No location data available for user {user_id}. Skipping alert.")
            return

        # ✅ AI Decision: Check if alert should be sent
        ai_decision = should_send_alert(user_id, live_lat, live_long)

        # ✅ Save decision to alert history
        new_alert = AlertHistory(
            user_id=user_id,
            latitude=live_lat,
            longitude=live_long,
            location_type="live",
            ai_decision=ai_decision,
        )
        db.session.add(new_alert)
        db.session.commit()

        # ❌ If AI says no alert needed, return
        if not ai_decision:
            print(f"🛑 AI decided NO alert needed for user {user_id}.")
            return

        # ✅ If AI says alert is needed, send email
        recipient_emails = tracking_users.get(user_id, {}).get("emails", [])
        google_maps_link = f"https://www.google.com/maps?q={live_lat},{live_long}"
        stop_tracking_link = f"https://phonelert-backend.onrender.com/stop-tracking?user_id={user_id}"

        subject = "🚨 Urgent: Your Phone is Still Left Behind!"
        body = f"""
        Your phone has not been retrieved yet. Please check its last known location!
        
        📍 **Last Known Location:** {google_maps_link}

        🏠 **Saved Location (if available):** {saved_location.name if saved_location else "Not Found"}

        🛑 **Stop Tracking:** Click here to stop alerts → [Stop Tracking]({stop_tracking_link})
        """

        for email in recipient_emails:
            try:
                msg = Message(subject, recipients=[email], body=body)
                mail.send(msg)
                print(f"✅ Email sent to {email}")
            except Exception as e:
                print(f"❌ Failed to send email to {email}: {str(e)}")
def send_repeated_alerts(user_id, recipient_emails):
    """Sends email alerts only if the phone remains in the same location for 3 minutes."""
    with app.app_context():
        phone_status = PhoneStatus.query.filter_by(user_id=user_id).first()

        if not phone_status:
            print(f"⚠️ No phone status found for user {user_id}.")
            return

        last_lat, last_long = phone_status.last_latitude, phone_status.last_longitude
        last_update_time = datetime.utcnow()

        while tracking_users.get(user_id, {}).get("active", False):
            time.sleep(180)  # ✅ Wait for 3 minutes

            phone_status = PhoneStatus.query.filter_by(user_id=user_id).first()
            if not phone_status:
                print(f"⚠️ No phone status found for user {user_id}. Stopping tracking.")
                break

            current_lat, current_long = phone_status.last_latitude, phone_status.last_longitude

            # ✅ Check if phone stayed in the same spot for 3 minutes
            if (current_lat, current_long) == (last_lat, last_long):
                print(f"📌 Phone has stayed in the same location for 3 minutes. Sending alert...")
                send_email_alert(user_id)  # ✅ Send email alert

            # ✅ Update last known position and timestamp
            last_lat, last_long = current_lat, current_long
            last_update_time = datetime.utcnow()



@app.route("/start-tracking", methods=["POST"])
def start_tracking():
    """Activates tracking only if the phone stays in one place for 3 minutes."""
    data = request.json
    print(f"📥 Received start-tracking request: {data}")
    sys.stdout.flush()  # ✅ Force log to appear in Render

    user_id = data.get("user_id")
    recipient_emails = data.get("emails", [])

    if not user_id or not recipient_emails:
        print("❌ Missing user_id or emails in request!")
        sys.stdout.flush()  # ✅ Force log to appear
        return jsonify({"error": "User ID and emails are required"}), 400

    if user_id in tracking_users and tracking_users[user_id]["active"]:
        print(f"⚠️ Tracking is already active for user {user_id}")
        sys.stdout.flush()  # ✅ Force log to appear
        return jsonify({"message": "Tracking is already active for this user"}), 200

    tracking_users[user_id] = {"active": True, "emails": recipient_emails}

    tracking_thread = threading.Thread(target=send_repeated_alerts, args=(user_id, recipient_emails), daemon=True)
    tracking_thread.start()

    print(f"🚀 Started tracking for user {user_id}")
    sys.stdout.flush()  # ✅ Force log to appear
    return jsonify({"message": "✅ Tracking started. If phone stays in one place for 3 minutes, an alert will be sent."}), 200







def monitor_phone_location(user_id):
    """Sends an email if the phone remains in the same location for 3 minutes."""
    with app.app_context():  # ✅ Ensure Flask context
        phone_status = PhoneStatus.query.filter_by(user_id=user_id).first()

        if not phone_status:
            print(f"⚠️ No phone status found for user {user_id}.")
            return

        last_lat, last_long = phone_status.last_latitude, phone_status.last_longitude
        last_update_time = datetime.now(timezone.utc)

        while tracking_users.get(user_id, {}).get("active", False):
            time.sleep(180)  # ✅ Wait for 3 minutes

            try:
                with db.session.begin():
                    phone_status = db.session.query(PhoneStatus).filter_by(user_id=user_id).first()

                if not phone_status:
                    print(f"⚠️ No phone status found for user {user_id}. Stopping tracking.")
                    break

                current_lat, current_long = phone_status.last_latitude, phone_status.last_longitude
                time_elapsed = (datetime.now(timezone.utc) - last_update_time).total_seconds()

                if (current_lat, current_long) == (last_lat, last_long) and time_elapsed >= 180:
                    print(f"📌 Phone has stayed in the same location for 3 minutes. Sending alert...")
                    send_email_alert(user_id)

                last_lat, last_long = current_lat, current_long
                last_update_time = datetime.now(timezone.utc)

            except OperationalError:
                print("🔄 Reconnecting to the database...")
                db.session.rollback()
                db.session.close()
                db.session = db.create_scoped_session()

        db.session.remove()  # ✅ Close session at the end











@app.route("/stop-tracking", methods=["GET", "POST"])
def stop_tracking():
    """Stops repeated email alerts when user clicks stop tracking link."""
    user_id = request.args.get("user_id") or request.json.get("user_id")
    print(f"📥 Received stop-tracking request for user {user_id}")
    sys.stdout.flush()  # ✅ Force log to appear

    if not user_id or user_id not in tracking_users:
        print("⚠️ Tracking was not active, ignoring stop request.")
        sys.stdout.flush()  # ✅ Force log to appear
        return jsonify({"error": "Tracking was not active for this user"}), 400

    tracking_users[user_id]["active"] = False
    del tracking_users[user_id]

    print(f"🛑 Stopped tracking for user {user_id}")
    sys.stdout.flush()  # ✅ Force log to appear
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
