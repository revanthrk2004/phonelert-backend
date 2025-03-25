import sys
import os
import logging

logging.basicConfig(level=logging.INFO)
import json
import time
import threading  # To run background tasks for AI detection
import requests  # To send notification to the user's other devices
# Force Python to recognize 'backend/' as a package
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

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
from dotenv import load_dotenv  # ✅ Load environment variables
from flask_jwt_extended import JWTManager
from routes.auth_route import auth


# ✅ Load .env file
load_dotenv()

app = create_app()
migrate = Migrate(app, db)  # ✅ Enable migrations
app.register_blueprint(auth, url_prefix='/auth')
# ✅ Configure Flask-Mail
app.config["MAIL_SERVER"] = "smtp.gmail.com"
app.config["MAIL_PORT"] = 587
app.config["MAIL_USE_TLS"] = True
app.config["MAIL_USERNAME"] = os.getenv("EMAIL_USER")  # ✅ Uses .env file
app.config["MAIL_PASSWORD"] = os.getenv("EMAIL_PASSWORD")  # ✅ Uses .env file
app.config["MAIL_DEFAULT_SENDER"] = os.getenv("EMAIL_USER")  # ✅ Uses your email as sender

mail = Mail(app)  # ✅ Initialize Flask-Mail


# ✅ Enable CORS for all requests
CORS(app, supports_credentials=True, resources={
    r"/*": {
        "origins": "http://localhost:8081",
        "methods": ["GET", "POST", "OPTIONS", "PUT", "DELETE"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})



knn_models = {}  # Store per-user trained models
tracking_users = {}  # ✅ Store tracking status per user
# ✅ Load API Key from Environment



def train_knn_model(user_id):
    with app.app_context():
        locations = UserLocation.query.filter_by(user_id=user_id).all()
        if not locations:
            print("⚠️ No locations to train for user:", user_id)
            return None

        X = []
        y = []

        for loc in locations:
            X.append([loc.latitude, loc.longitude])
            y.append(1 if loc.location_type == "safe" else 0)

        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(np.array(X), np.array(y))
        knn_models[user_id] = model
        print(f"✅ Trained KNN for user {user_id} on {len(X)} locations.")
        return model

def predict_location_safety(user_id, latitude, longitude):
    if user_id not in knn_models:
        print("🧠 No model found. Training KNN on the fly...")
        model = train_knn_model(user_id)
        if not model:
            return "unknown"
    else:
        model = knn_models[user_id]

    prediction = model.predict([[latitude, longitude]])[0]
    return "safe" if prediction == 1 else "unsafe"


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


@app.route("/update-phone-status", methods=["POST"])
def update_phone_status():
    data = request.json
    user_id = data.get("user_id")
    latitude = data.get("latitude")
    longitude = data.get("longitude")

    if not user_id or latitude is None or longitude is None:
        return jsonify({"error": "Missing user_id or coordinates"}), 400

    phone_status = PhoneStatus.query.filter_by(user_id=user_id).first()
    if not phone_status:
        phone_status = PhoneStatus(user_id=user_id)

    phone_status.last_latitude = latitude
    phone_status.last_longitude = longitude
    phone_status.tracking_active = True

    db.session.add(phone_status)
    db.session.commit()

    print(f"📡 Updated phone status: ({latitude}, {longitude}) for user {user_id}")
    return jsonify({"message": "Phone location updated"}), 200



@app.route("/live-location", methods=["POST"])
def live_location():
    data = request.json
    user_id = data.get("user_id")
    latitude = data.get("latitude")
    longitude = data.get("longitude")
    emails = data.get("emails")

    if not user_id or latitude is None or longitude is None or not emails:
        return jsonify({"error": "Missing data"}), 400

    # 🧠 Let AI decide if it's unsafe and if an alert should be sent
    ai_decision = ai_decide_alert(user_id, latitude, longitude)

    if ai_decision == "unsafe":
        send_email_alert(user_id, emails, latitude, longitude)
        return jsonify({"message": "Alert sent", "ai_decision": ai_decision})
    else:
        return jsonify({"message": "No alert needed", "ai_decision": ai_decision})



@app.route("/check-location", methods=["POST"])
def check_location():
    """Receives live location data and sends an alert with the latest coordinates."""
    data = request.json
    print("📥 Incoming check-location request:", data)  # 🔍 LOG the incoming data
    location_name = data.get("locationName", "Live Location")
    user_id = data.get("user_id")
    latitude = data.get("latitude")  # ✅ Get real-time latitude
    longitude = data.get("longitude")  # ✅ Get real-time longitude
    recipient_emails = data.get("emails", [])  
    
    if not user_id or latitude is None or longitude is None or not recipient_emails:
        print("❌ Missing data in request:", {
            "user_id": user_id,
            "latitude": latitude,
            "longitude": longitude,
            "emails": recipient_emails
        })
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










def classify_location_by_ai(user_id, latitude, longitude):
    with app.app_context():
        user_locations = UserLocation.query.filter_by(user_id=user_id).all()

        if not user_locations:
            print("⚠️ No location history found for user.")
            return "unknown"

        current_coords = (latitude, longitude)

        for loc in user_locations:
            loc_coords = (loc.latitude, loc.longitude)
            distance = geodesic(current_coords, loc_coords).meters

            print(f"📏 Distance from ({loc.latitude}, {loc.longitude}) → {distance:.2f}m")

            if distance <= loc.radius:
                print(f"✅ Inside radius {loc.radius}m → Classified as {loc.location_type}")
                return loc.location_type

        print("❌ Not within any saved location radius. Marking as unsafe.")
        return "unsafe"


def send_email_alert(user_id, recipient_emails, live_lat=None, live_long=None):
    """Sends email alert based on stored location type (safe/unsafe)."""
    with app.app_context():
        if live_lat is None or live_long is None:
            phone_status = PhoneStatus.query.filter_by(user_id=user_id).first()
        if phone_status:
                live_lat, live_long = phone_status.last_latitude, phone_status.last_longitude

        location_type = classify_location_by_ai(user_id, live_lat, live_long)
        print(f"🧠 AI classified location as: {location_type}")

    if location_type == "unknown":
        print("🤖 AI: Unknown location — letting frontend ask the user to label.")
        return jsonify({
            "message": "unknown_location",
            "latitude": latitude,
            "longitude": longitude
        }), 200

    if location_type.lower() == "safe":
        print("✅ AI: Safe location — skipping alert.")
        return

# Only continue if AI says unsafe


        # 📨 Now this alert code runs only for "unsafe"
        google_maps_link = f"https://www.google.com/maps?q={live_lat},{live_long}"
        stop_tracking_link = f"https://phonelert-backend.onrender.com/stop-tracking?user_id={user_id}"

        subject = "🚨 Alert: Your Phone Might Be Left Behind!"
        body = f"""
        Your phone has stayed too long at a potentially unsafe location.

        📍 Location: {google_maps_link}
        🛑 Stop Tracking: {stop_tracking_link}
        """

        for email in recipient_emails:
            try:
                msg = Message(subject, recipients=[email], body=body)
                mail.send(msg)
                print(f"✅ Alert Email sent to {email}")
            except Exception as e:
                print(f"❌ Failed to send email to {email}: {str(e)}")




@app.route("/add-location", methods=["POST"])
def add_location():
    data = request.json
    user_id = data.get("user_id")
    location_name = data.get("name")
    latitude = data.get("latitude")
    longitude = data.get("longitude")
    location_type = data.get("location_type")

    if not user_id or not location_name or latitude is None or longitude is None or not location_type:
        return jsonify({"error": "Missing data"}), 400

    try:
        new_location = UserLocation(
            user_id=user_id,
            location_name=location_name,
            latitude=latitude,
            longitude=longitude,
            location_type=location_type,
            visible=True  # 🧠 MAKE SURE THIS IS SET
        )
        db.session.add(new_location)
        db.session.commit()
        print(f"✅ Location saved: {location_name} ({location_type}) for user {user_id}")

        return jsonify({"message": "Location saved successfully"}), 200
    except Exception as e:
        print(f"❌ Error saving location: {e}")
        return jsonify({"error": str(e)}), 500




def send_repeated_alerts(user_id, recipient_emails):
    """AI-powered alerts: Only sends emails if AI confirms it's necessary."""
    with app.app_context():
        try:
            # ✅ Ensure user_id is an integer
            user_id = int(user_id)
        except (ValueError, TypeError):
            print(f"❌ Invalid user_id received in send_repeated_alerts: {user_id}")
            return

        # ✅ Fetch phone status from database
        phone_status = PhoneStatus.query.filter_by(user_id=user_id).first()
        if not phone_status:
            print(f"⚠️ No phone status found for user {user_id}. Stopping tracking.")
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
                print(f"📌 Phone has stayed in the same location for 3 minutes. Asking AI for a decision...")

                # ✅ Ask AI to make a decision
                decision = ai_decide_alert(user_id, current_lat, current_long)
                if decision == "unsafe":
                    print("🚨 AI says UNSAFE. Sending alert...")
                    send_email_alert(user_id, recipient_emails, current_lat, current_long)
                elif decision == "safe":
                    print("✅ AI says SAFE. No alert needed.")
                else:
                    print("🤔 AI returned 'no_alert' or unknown. Skipping this round.")

                # ✅ Update last known position and timestamp
                last_lat, last_long = current_lat, current_long
                last_update_time = datetime.utcnow()





@app.route("/start-tracking", methods=["POST"])
def start_tracking():
    """Activates tracking only if the phone stays in one place for 3 minutes."""
    data = request.json
    print(f"📥 Received start-tracking request: {data}")  
    sys.stdout.flush()

    # ✅ Log request headers for debugging
    print(f"🧐 Request Headers: {request.headers}")
    sys.stdout.flush()

    recipient_emails = data.get("emails", [])
    user_id = data.get("user_id")

    if not user_id:
        print("❌ Missing user_id in request!")
        sys.stdout.flush()
        return jsonify({"error": "User ID is required"}), 400

    print(f"🔍 Processing user_id: {user_id} (Type: {type(user_id)})")
    sys.stdout.flush()

    if user_id in tracking_users and tracking_users[user_id]["active"]:
        print(f"⚠️ Tracking is already active for user {user_id}")
        sys.stdout.flush()
        return jsonify({"message": "Tracking is already active for this user"}), 200

    tracking_users[user_id] = {"active": True, "emails": recipient_emails}

    tracking_thread = threading.Thread(target=send_repeated_alerts, args=(user_id, recipient_emails), daemon=True)
    tracking_thread.start()

    print(f"🚀 Started tracking for user {user_id}")
    sys.stdout.flush()
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




def ai_decide_alert(user_id, latitude, longitude):
    """AI decides whether an alert should be sent based on location history."""
    with app.app_context():
        # ✅ Check if this location is already classified
        user_locations = UserLocation.query.filter_by(user_id=user_id).all()

        if not user_locations:
            print("⚠️ No visible locations to learn from.")
            return "no_alert"

        current_coords = (latitude, longitude)
        closest_location = None
        min_distance = float("inf")

        for loc in user_locations:
            loc_coords = (loc.latitude, loc.longitude)
            distance = geodesic(current_coords, loc_coords).meters

            if distance <= loc.radius and distance < min_distance:
                closest_location = loc
                min_distance = distance

        if closest_location:
            print(f"🧠 AI Decision: Closest match → {closest_location.location_name} ({closest_location.location_type})")
            location_type = closest_location.location_type
        else:
            print("❌ AI Decision: No match found. Marking as unknown.")
            location_type = "unknown"

        # ✅ Check phone status to confirm it's stationary
        phone_status = PhoneStatus.query.filter_by(user_id=user_id).first()
        if phone_status and (phone_status.last_latitude, phone_status.last_longitude) == (latitude, longitude):
            print("📌 Phone is stationary. Logging alert history.")

            # ✅ Log the AI decision in `alert_history`
            new_alert = AlertHistory(
                user_id=user_id,
                latitude=latitude,
                longitude=longitude,
                location_type=location_type,
                ai_decision="sent"
            )
            db.session.add(new_alert)
            db.session.commit()

            return location_type

        return "no_alert"


@app.route('/get-locations/<int:user_id>', methods=['GET'])
def get_locations(user_id):
    locations = UserLocation.query.filter_by(user_id=user_id).all()
    return jsonify([{
        "name": loc.location_name,
        "latitude": loc.latitude,
        "longitude": loc.longitude,
        "location_type": loc.location_type
    } for loc in locations]), 200


@app.route("/soft-delete-location", methods=["POST"])
def soft_delete_location():
    data = request.json
    user_id = data.get("user_id")
    latitude = data.get("latitude")
    longitude = data.get("longitude")

    if not user_id or latitude is None or longitude is None:
        return jsonify({"error": "Missing data"}), 400

    location = UserLocation.query.filter_by(user_id=user_id, latitude=latitude, longitude=longitude).first()
    if not location:
        return jsonify({"error": "Location not found"}), 404

    location.visible = False
    db.session.commit()
    return jsonify({"message": "Location soft-deleted (hidden from UI)"}), 200





@app.route("/ai-location-check", methods=["POST"])
def ai_location_check():
    """Returns AI classification and number of learned locations."""
    data = request.json
    user_id = data.get("user_id")
    latitude = data.get("latitude")
    longitude = data.get("longitude")

    if not user_id or latitude is None or longitude is None:
        return jsonify({"error": "Missing user_id or coordinates"}), 400

    try:
    user_locations = UserLocation.query.filter_by(user_id=user_id).all()

    current_coords = (latitude, longitude)
    is_safe = False
    used_locations = []

    # ✅ Collect nearby locations (within 200m)
    for loc in user_locations:
        loc_coords = (loc.latitude, loc.longitude)
        distance = geodesic(current_coords, loc_coords).meters

        if distance <= 200:
            used_locations.append({
                "name": loc.location_name,
                "type": loc.location_type,
                "distance": distance
            })

    # ✅ Use trained AI to decide
    ai_decision = predict_location_safety(user_id, latitude, longitude)
    is_safe = (ai_decision == "safe")

    return jsonify({
        "is_safe": is_safe,
        "used_locations": used_locations,
        "total_learned": len(user_locations),
        "ai_decision": ai_decision
    }), 200






@app.route("/stop-tracking", methods=["GET", "POST"])
def stop_tracking():
    """Stops repeated email alerts when user clicks stop tracking link."""
    
    user_id = request.args.get("user_id") or (request.json.get("user_id") if request.is_json else None)
    print(f"📥 Received stop-tracking request for user {user_id}")
    sys.stdout.flush()

    if not user_id:
        print("❌ Missing user_id in request!")
        sys.stdout.flush()
        return jsonify({"error": "User ID is required"}), 400

    if user_id not in tracking_users:
        print(f"⚠️ No active tracking found for user {user_id}")
        sys.stdout.flush()
        return jsonify({"error": "Tracking was not active for this user"}), 400

    tracking_users[user_id]["active"] = False
    del tracking_users[user_id]

    print(f"🛑 Stopped tracking for user {user_id}")
    sys.stdout.flush()
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
