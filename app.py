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
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

import numpy as np





from flask_mail import Mail, Message  # ✅ Add Flask-Mail
from database.models import AlertHistory  # ✅ Add this line

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from flask import Flask, request, jsonify
from flask_cors import CORS
from database.db_manager import create_app, db
from routes.auth_route import auth
from database.models import PhoneStatus, UserLocation  # ✅ Import UserLocation

from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut

from flask_sqlalchemy import SQLAlchemy  # Database to store locations
from geopy.distance import geodesic  # To calculate distance between two coordinates
from datetime import datetime, timedelta  # For timestamping last phone activity
from database.models import PhoneStatus  # ✅ Import the model
from flask_migrate import Migrate
from sqlalchemy import inspect
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy.exc import OperationalError
from dotenv import load_dotenv  # ✅ Load environment variables
from flask_jwt_extended import JWTManager
from routes.auth_route import auth
from datetime import timezone

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
        "origins": "https://phonelert-backend.onrender.com",
        "methods": ["GET", "POST", "OPTIONS", "PUT", "DELETE"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})



knn_models = {}  # Store per-user trained models
tracking_users = {}  # ✅ Store tracking status per user
# ✅ Load API Key from Environment




def geocode_with_retry(geolocator, location_str, retries=3, delay=2):
    for i in range(retries):
        try:
            return geolocator.geocode(location_str, timeout=5)
        except GeocoderTimedOut:
            print(f"⏳ Timeout on '{location_str}', retrying in {delay}s...")
            time.sleep(delay)
            delay *= 2  # Exponential backoff
        except Exception as e:
            print(f"❌ Geocoding failed: {e}")
            break
    return None


def is_anomalous_location(user_id, latitude, longitude, threshold=100):
    """
    Detect if the given location is far from known clusters.
    threshold: max distance in meters from any cluster center
    """
    with app.app_context():
        history = get_user_location_history(user_id)
        if not history or len(history) < 3:
            print("⚠️ Not enough data to perform anomaly detection.")
            return False  # Not enough to judge

        coords = np.array(history)

        # Train a KMeans model (choose small number of clusters)
        kmeans = KMeans(n_clusters=min(3, len(coords)))
        kmeans.fit(coords)

        cluster_centers = kmeans.cluster_centers_

        current_point = (latitude, longitude)

        for center in cluster_centers:
            center_point = (center[0], center[1])
            distance = geodesic(current_point, center_point).meters

            if distance < threshold:
                return False  # It's close to a known cluster

        print(f"🚨 Anomaly Detected! Distance from clusters > {threshold}m")
        return True  # Too far from all clusters → anomaly



def cluster_and_save_user_locations(user_id, eps=50, min_samples=2):
    """Cluster historical alert locations and save new UserLocation entries."""
    from database.models import UserLocation, AlertHistory
    from geopy.distance import geodesic
    import numpy as np

    with app.app_context():
        history = get_user_location_history(user_id)
        if not history:
            print(f"⚠️ No alert history found for user {user_id}.")
            return

        coords = np.array(history)
        kms_per_radian = 6371.0088
        epsilon = eps / 1000.0 / kms_per_radian  # meters to radians

        dbscan = DBSCAN(eps=epsilon, min_samples=min_samples, metric='haversine')
        labels = dbscan.fit_predict(np.radians(coords))

        cluster_centers = {}
        for label in set(labels):
            if label == -1:
                continue  # noise
            cluster_points = coords[labels == label]
            center = cluster_points.mean(axis=0)
            cluster_centers[label] = center

        saved_count = 0
        for label, (lat, lon) in cluster_centers.items():
            lat = float(lat)
            lon = float(lon)
            location_name = f"Cluster {label + 1}"
            existing_locations = UserLocation.query.filter_by(user_id=user_id, visible=True).all()

            too_close = False
            for existing in existing_locations:
                dist = geodesic((lat, lon), (existing.latitude, existing.longitude)).meters
                if dist < 30:
                    too_close = True
                    break

            if not too_close:
                new_loc = UserLocation(
                    user_id=user_id,
                    latitude=lat,
                    longitude=lon,
                    location_name=location_name,
                    location_type="unsafe",
                    radius=50,
                    visible=True
                )
                db.session.add(new_loc)
                saved_count += 1
                print(f"✅ Saved cluster location: {location_name} at ({lat}, {lon})")
            else:
                print(f"⏩ Skipped {location_name} (too close to existing location)")

        db.session.commit()
        print(f"🏁 Clustering complete. {saved_count} new locations saved for user {user_id}.")



def is_near_cluster(user_id, current_lat, current_long, threshold_meters=50):
    """Check if current location is near a learned AI cluster."""
    from geopy.distance import geodesic

    clusters = find_location_clusters(user_id)
    if not clusters:
        return False, None  # No learned spots

    current = (current_lat, current_long)
    for center in clusters:
        distance = geodesic(current, center).meters
        if distance <= threshold_meters:
            print(f"🤖 Matched AI cluster at {center} [~{distance:.2f}m away]")
            return True, center

    return False, None


def find_location_clusters(user_id, eps=0.0005, min_samples=3):
    """
    Cluster user's location history using DBSCAN.
    eps ~ approx 50m, adjust based on density.
    """
    coords = get_user_location_history(user_id)
    if not coords:
        print("⚠️ No history to cluster for user", user_id)
        return []

    # Convert to numpy array
    X = np.array(coords)

    # Run DBSCAN
    db = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
    labels = db.fit_predict(X)

    clusters = {}
    for label, (lat, long) in zip(labels, coords):
        if label == -1:
            continue  # skip noise
        if label not in clusters:
            clusters[label] = []
        clusters[label].append((lat, long))

    # Get cluster centers (mean of points)
    cluster_centers = []
    for points in clusters.values():
        lats, longs = zip(*points)
        center = (sum(lats) / len(lats), sum(longs) / len(longs))
        cluster_centers.append(center)

    print(f"✅ Found {len(cluster_centers)} clusters for user {user_id}")
    return cluster_centers

def get_user_location_history(user_id):
    """Fetch past coordinates from alert history for clustering."""
    with app.app_context():
        records = AlertHistory.query.filter_by(user_id=user_id).all()
        return [(r.latitude, r.longitude) for r in records]






def train_knn_model(user_id, force=False):  # ✅ Add force argument
    with app.app_context():
        locations = UserLocation.query.filter_by(user_id=user_id, visible=True).all()
        if not locations:
            print("⚠️ No locations to train for user:", user_id)
            return None

        existing = knn_models.get(user_id)
        if not force and existing and len(locations) == existing["n_samples"]:
            print("📦 Skipping retrain — model already up to date")
            return existing["model"]

        X = [[loc.latitude, loc.longitude] for loc in locations]
        y = [1 if loc.location_type == "safe" else 0 for loc in locations]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(X_scaled, y)

        knn_models[user_id] = {
            "model": model,
            "scaler": scaler,
            "n_samples": len(locations)  # ✅ Save training size here
        }

        print(f"✅ Trained KNN for user {user_id} on {len(X)} locations.")
        return model


def predict_location_safety(user_id, latitude, longitude):
    if user_id not in knn_models:
        print("🧠 No model found. Training KNN on the fly...")
        train_knn_model(user_id)
        if user_id not in knn_models:
            return "unknown"

    model_data = knn_models[user_id]
    model = model_data["model"]
    scaler = model_data["scaler"]

    coords_scaled = scaler.transform([[latitude, longitude]])
    prediction = model.predict(coords_scaled)[0]
    return "safe" if prediction == 1 else "unsafe"







@app.route('/news-sentiment', methods=['GET'])
def news_sentiment():
    query = request.args.get('query')
    if not query:
        return jsonify({"error": "Missing query"}), 400

    GNEWS_API_KEY = os.getenv("GNEWS_API_KEY")
    url = f"https://gnews.io/api/v4/search?q={query}&lang=en&token={GNEWS_API_KEY}"

    try:
        res = requests.get(url)
        res.raise_for_status()
        articles = res.json().get("articles", [])

        results = []
        for article in articles:
            results.append({
                "title": article.get("title"),
                "url": article.get("url"),
                "description": article.get("description"),
                "publishedAt": article.get("publishedAt")
            })

        return jsonify({
            "query": query,
            "total_results": len(results),
            "articles": results
        }), 200

    except requests.exceptions.RequestException as e:
        print("❌ News API failed:", str(e))
        return jsonify({"error": "News API failed", "details": str(e)}), 500


@app.route("/local-news", methods=["GET"])
def fetch_local_news():
    city = request.args.get("area", "London")
    keywords = ["robbery", "theft", "crime", "unsafe", "danger", "stabbing", "mugging"]
    GNEWS_API_KEY = os.getenv("GNEWS_API_KEY")

    try:
        query = " OR ".join([f'"{word} in {city}"' for word in keywords])
        url = f"https://gnews.io/api/v4/search?q={query}&lang=en&max=10&token={GNEWS_API_KEY}"
        res = requests.get(url)
        res.raise_for_status()
        articles = res.json().get("articles", [])

        flagged = []
        geolocator = Nominatim(user_agent="phonelert-news-locator")

        for article in articles:
            title = article.get("title", "")
            guess = title.split("in")[-1].strip().split(" ")[0:2]
            location_str = " ".join(guess) + f", {city}"
            time.sleep(1)
            loc = geocode_with_retry(geolocator, location_str, retries=3, delay=2)
            if loc:
                flagged.append({
                    "title": title,
                    "location": location_str,
                    "lat": loc.latitude,
                    "lon": loc.longitude
                })

                # ✅ Check if already exists to prevent duplicates
                existing = UserLocation.query.filter_by(
                    user_id=11,
                    latitude=loc.latitude,
                    longitude=loc.longitude
                ).first()

                if existing:
                    print(f"⏩ Skipping duplicate location: {location_str}")
                    continue

                unsafe = UserLocation(
                    user_id=11,
                    location_name=location_str,
                    latitude=loc.latitude,
                    longitude=loc.longitude,
                    location_type="unsafe",
                    visible=False,
                    radius=50,
                    timestamp=datetime.now(timezone.utc)
                )
                db.session.add(unsafe)

        db.session.commit()
        return jsonify({"city": city, "flagged": flagged}), 200

    except Exception as e:
        print("❌ News fetch failed:", str(e))
        return jsonify({"error": str(e)}), 500




@app.route("/diagnose-ai/<int:user_id>", methods=["GET"])
def diagnose_ai(user_id):
    with app.app_context():
        locations = UserLocation.query.filter_by(user_id=user_id, visible=True).all()
        if not locations:
            return jsonify({"error": "No data to evaluate"}), 400

        safe_count = sum(1 for loc in locations if loc.location_type == "safe")
        unsafe_count = sum(1 for loc in locations if loc.location_type == "unsafe")
        clusters = find_location_clusters(user_id)

        accuracy = None
        mismatches = "N/A"

        if user_id in knn_models and "scaler" in knn_models[user_id] and "model" in knn_models[user_id]:

            model_data = knn_models[user_id]
            X = [[loc.latitude, loc.longitude] for loc in locations]
            y_true = [1 if loc.location_type == "safe" else 0 for loc in locations]
            X_scaled = model_data["scaler"].transform(X)
            y_pred = model_data["model"].predict(X_scaled)
            accuracy = accuracy_score(y_true, y_pred)
            mismatches = int(sum(a != b for a, b in zip(y_true, y_pred))) if accuracy is not None else "N/A"

        return jsonify({
            "total_locations": len(locations),
            "safe_locations": safe_count,
            "unsafe_locations": unsafe_count,
            "cluster_count": len(clusters),
            "model_trained": user_id in knn_models,
            
            "mismatches": mismatches,
            "accuracy_estimate": round(accuracy, 4) if accuracy is not None else "Model not trained"
            
        }), 200


@app.route("/evaluate-classification", methods=["GET"])
def evaluate_classification_model():
    """Evaluate AI decision logic using precision, recall, and F1."""
    user_id = request.args.get("user_id", type=int)
    if not user_id:
        return jsonify({"error": "user_id is required"}), 400

    with app.app_context():
        user_locations = UserLocation.query.filter_by(user_id=user_id, visible=True).all()
        if not user_locations:
            return jsonify({"error": "No saved locations found for this user"}), 404

        X = []
        y_true = []

        for loc in user_locations:
            X.append([loc.latitude, loc.longitude])
            y_true.append(1 if loc.location_type == "unsafe" else 0)

        y_pred = []
        for coords in X:
            pred = predict_location_safety(user_id, coords[0], coords[1])
            y_pred.append(1 if pred == "unsafe" else 0)

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        accuracy = sum(1 for a, b in zip(y_true, y_pred) if a == b) / len(y_true)

        return jsonify({
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
        }), 200


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

    try:
        phone_status = PhoneStatus.query.filter_by(user_id=user_id).first()
        if phone_status:
            phone_status.last_latitude = latitude
            phone_status.last_longitude = longitude
        else:
            phone_status = PhoneStatus(
                user_id=user_id,
                last_latitude=latitude,
                last_longitude=longitude
            )
            db.session.add(phone_status)

        db.session.commit()
        print(f"📲 Phone status updated for user {user_id}: {latitude}, {longitude}")
        return jsonify({"message": "Phone status updated"}), 200

    except Exception as e:
        print("❌ Error updating phone status:", e)
        return jsonify({"error": "Failed to update phone status", "details": str(e)}), 500



@app.route("/live-location", methods=["POST"])
def live_location():
    data = request.json
    user_id = data.get("user_id")
    latitude = data.get("latitude")
    longitude = data.get("longitude")
    emails = data.get("emails")
    force_alert = data.get("force_alert", False)  # 🆕 NEW LINE
    
    if not user_id or latitude is None or longitude is None or not emails:
        return jsonify({"error": "Missing data"}), 400

    # 🧠 Let AI decide if it's unsafe and if an alert should be sent
    if force_alert:
        ai_decision = "unsafe"  # 🔥 Force it for shake case
        print("🛑 Force alert enabled — skipping AI checks!")
    else:
        ai_decision = ai_decide_alert(user_id, latitude, longitude)

    if ai_decision == "unsafe":
        send_email_alert(user_id, emails, latitude, longitude, force_alert=force_alert)
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


def send_email_alert(user_id, recipient_emails, live_lat=None, live_long=None, force_alert=False):
    """Sends email alert based on stored location type (safe/unsafe)."""
    with app.app_context():
        phone_status = None
        if live_lat is None or live_long is None:
            phone_status = PhoneStatus.query.filter_by(user_id=user_id).first()
            if phone_status:
                live_lat, live_long = phone_status.last_latitude, phone_status.last_longitude

        location_type = classify_location_by_ai(user_id, live_lat, live_long)

        # 🚨 Force override
        if force_alert:
            print("🛑 Force alert enabled — skipping AI checks!")
            location_type = "unsafe"

        print(f"🧠 AI classified location as: {location_type}")

        if location_type == "unknown":
            print("🤖 AI: Unknown location — letting frontend ask the user to label.")
            return jsonify({
                "message": "unknown_location",
                "latitude": live_lat,
                "longitude": live_long
            }), 200

        if location_type.lower() == "safe" and not force_alert:
            print("✅ AI: Safe location — skipping alert.")
            return

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
        # 🔍 Check if this location already exists (even if soft-deleted)
        existing_location = UserLocation.query.filter_by(
            user_id=user_id,
            latitude=latitude,
            longitude=longitude
        ).first()

        if existing_location:
            # 🔁 Update instead of adding new
            existing_location.visible = True
            existing_location.location_name = location_name
            existing_location.location_type = location_type
            existing_location.timestamp = datetime.now(timezone.utc)
            print(f"♻️ Reactivated location: {location_name} for user {user_id}")
        else:
            # 🆕 Add new location
            new_location = UserLocation(
                user_id=user_id,
                location_name=location_name,
                latitude=latitude,
                longitude=longitude,
                location_type=location_type,
                radius=50,
                timestamp=datetime.now(timezone.utc),
                visible=True
            )
            db.session.add(new_location)
            print(f"✅ New location added: {location_name} for user {user_id}")

        db.session.commit()

         # ✅ AUTO-TRAIN AI MODEL
        train_knn_model(user_id, force=True)
        print(f"🤖 Auto-trained KNN model for user {user_id}")

        return jsonify({"message": "Location saved and AI model retrained"}), 200

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
        last_update_time = datetime.now(timezone.utc)

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
                last_update_time = datetime.now(timezone.utc)





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
    with app.app_context():
        user_locations = UserLocation.query.filter_by(user_id=user_id, visible=True).all()
        if not user_locations:
            print("⚠️ No visible locations to learn from.")
            return "no_alert"

        current_coords = (latitude, longitude)
        buffer = 20  # meters
        hour = datetime.now(timezone.utc).hour

        # 1. Match saved locations
        sorted_locations = sorted(
            user_locations,
            key=lambda loc: geodesic(current_coords, (loc.latitude, loc.longitude)).meters
        )

        closest_location = None
        for loc in sorted_locations:
            dist = geodesic(current_coords, (loc.latitude, loc.longitude)).meters
            if dist <= loc.radius + buffer:
                closest_location = loc
                break

        # 2. AI Location type logic
        if closest_location:
            location_type = closest_location.location_type
            print(f"🧠 Closest match → {closest_location.location_name} ({location_type})")
        else:
            matched_cluster, _ = is_near_cluster(user_id, latitude, longitude)
            if matched_cluster:
                location_type = "unsafe"
                print("🤖 Near learned cluster")
            elif is_anomalous_location(user_id, latitude, longitude):
                location_type = "anomaly"
                print("🚨 AI detected anomaly")
            else:
                location_type = "unknown"
                print("❌ No matches — location marked unknown")

        # 3. Suppress duplicate alerts (same spot in 10 mins), but NOT if user marked as unsafe
            recent_alerts = AlertHistory.query.filter(
                AlertHistory.user_id == user_id,
                AlertHistory.latitude == latitude,
                AlertHistory.longitude == longitude,
                AlertHistory.timestamp >= datetime.now(timezone.utc) - timedelta(minutes=10)
            ).count()
            if recent_alerts >= 100:
                print("🛑 Duplicate alert suppressed (not unsafe area)")
                return "no_alert"


        # 4. Risk Score (based on time)
        risk_score = 0.1
        if location_type in ["unsafe", "unknown", "anomaly"]:
            if 1 <= hour <= 5:
                risk_score = 1.0
            elif 22 <= hour or hour <= 6:
                risk_score = 0.8
            elif 9 <= hour <= 17:
                risk_score = 0.3
            else:
                risk_score = 0.5

        # 5. Save alert only if phone is stationary
        phone_status = PhoneStatus.query.filter_by(user_id=user_id).first()
        if phone_status:
            phone_coords = (phone_status.last_latitude, phone_status.last_longitude)
            if geodesic(phone_coords, current_coords).meters < 20:
                is_anomaly = False
                reason = None
                recent_count = AlertHistory.query.filter(
                    AlertHistory.user_id == user_id,
                    AlertHistory.timestamp >= datetime.now(timezone.utc) - timedelta(minutes=5)
                ).count()

                if recent_count >= 5:
                    is_anomaly = True
                    reason = "Too many alerts in short time"
                elif 1 <= hour <= 4:
                    is_anomaly = True
                    reason = "Alert during late-night hours"

                new_alert = AlertHistory(
                    user_id=user_id,
                    latitude=latitude,
                    longitude=longitude,
                    location_type=location_type,
                    ai_decision="sent",
                    was_anomaly=is_anomaly,
                    reason=reason,
                    risk_score=risk_score
                )
                db.session.add(new_alert)
                db.session.commit()

                if location_type in ["unsafe", "anomaly"] and phone_status.expo_push_token:
                    try:
                        requests.post(
                            "https://exp.host/--/api/v2/push/send",
                            headers={"Accept": "application/json", "Content-Type": "application/json"},
                            json={
                                "to": phone_status.expo_push_token,
                                "sound": "default",
                                "title": "⚠️ PHONELERT ALERT",
                                "body": f"Phone detected in a {location_type.upper()} area.",
                            }
                        )
                        print("📳 Push sent.")
                    except Exception as e:
                        print("❌ Push failed:", e)

                alert_total = AlertHistory.query.filter_by(user_id=user_id).count()
                if alert_total % 10 == 0:
                    print("🤖 Clustering triggered (10 alerts)")
                    cluster_and_save_user_locations(user_id)

                print(f"📊 Risk Score: {risk_score}")
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
    data = request.json
    user_id = data.get("user_id")
    latitude = data.get("latitude")
    longitude = data.get("longitude")

    if not user_id or latitude is None or longitude is None:
        return jsonify({"error": "Missing user_id or coordinates"}), 400

    try:
        user_locations = UserLocation.query.filter_by(user_id=user_id, visible=True).all()
        current_coords = (latitude, longitude)
        used_locations = []

        closest_location = None
        closest_distance = float("inf")

        for loc in user_locations:
            loc_coords = (loc.latitude, loc.longitude)
            distance = geodesic(current_coords, loc_coords).meters
            if distance <= 200:
                used_locations.append({
                    "name": loc.location_name,
                    "type": loc.location_type,
                    "distance": distance,
                    "latitude": loc.latitude,
                    "longitude": loc.longitude
                })
                if distance < closest_distance:
                    closest_distance = distance
                    closest_location = loc

        if closest_location:
            is_safe = (closest_location.location_type == "safe")
            ai_decision = closest_location.location_type
            print(f"📍 Nearby match: {closest_location.location_name} ({ai_decision}) [{closest_distance:.2f}m]")
        else:
            ai_decision = predict_location_safety(user_id, latitude, longitude)
            is_safe = (ai_decision == "safe")
            print(f"❌ AI fallback decision: {ai_decision}")

            if ai_decision == "unknown" and user_locations:
                print("🔁 Retraining AI model for user", user_id)
                retrain_model_for_user(user_id)

        return jsonify({
            "is_safe": is_safe,
            "used_locations": used_locations,
            "total_learned": len(user_locations),
            "ai_decision": ai_decision
        }), 200

    except Exception as e:
        print("❌ Error in ai_location_check:", e)
        return jsonify({"error": "AI check failed", "details": str(e)}), 500


def retrain_model_for_user(user_id):
    print(f"🧠 [Retrain] Forcing model retrain for user {user_id}")
    train_knn_model(user_id, force=True)





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

    try:
        user_id = int(user_id)  # 🧠 Convert to int (IMPORTANT)
    except ValueError:
        print("❌ Invalid user_id format")
        sys.stdout.flush()
        return jsonify({"error": "Invalid user_id format"}), 400    

    if user_id not in tracking_users:
        print(f"⚠️ No active tracking found for user {user_id}")
        sys.stdout.flush()
        return jsonify({
            "message": "⚠️ Tracking was already inactive",
            "stop_tracking": True,
            "stop_shake_alert": True  # ✅ Always send stop_shake_alert now
        }), 200

    tracking_users[user_id]["active"] = False
    del tracking_users[user_id]

    print(f"🛑 Stopped tracking for user {user_id}")
    sys.stdout.flush()
    return jsonify({
        "message": "✅ Tracking stopped successfully",
        "stop_tracking": True  
        "stop_shake_alert": True  
    }), 200





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






def send_push_notification(push_token, title, body):
    """Send push notification to the user's device."""
    message = {
        'to': push_token,
        'sound': 'default',
        'title': title,
        'body': body,
        'priority': 'high',
    }

    response = requests.post("https://exp.host/--/api/v2/push/send", json=message)
    print("📬 Expo Push Notification Response:", response.status_code, response.text)


@app.route('/test-push/<int:user_id>', methods=['GET'])
def test_push(user_id):
    phone_status = PhoneStatus.query.filter_by(user_id=user_id).first()
    if not phone_status or not phone_status.expo_push_token:
        return jsonify({"error": "No push token for user"}), 400

    send_push_notification(
        phone_status.expo_push_token,
        "🔧 Manual Test",
        "This is a test push from backend."
    )
    return jsonify({"message": "Push sent"})



@app.route('/save-push-token', methods=['POST'])
def save_push_token():
    data = request.get_json()
    user_id = data.get("user_id")
    token = data.get("expo_push_token")

    if not user_id or not token:
        return jsonify({"error": "Missing user_id or expo_push_token"}), 400

    with app.app_context():
        phone_status = PhoneStatus.query.filter_by(user_id=user_id).first()
        if not phone_status:
            phone_status = PhoneStatus(user_id=user_id)

        phone_status.expo_push_token = token
        db.session.add(phone_status)
        db.session.commit()

    print(f"📱 Push token saved for user {user_id}: {token}")
    return jsonify({"message": "Push token saved successfully!"})




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


@app.route("/cluster-user-locations/<int:user_id>", methods=["GET"])
def cluster_user_locations(user_id):
    try:
        cluster_and_save_user_locations(user_id)
        return jsonify({"message": "✅ Zones Updated"}), 200  # 🔧 Unified message for frontend
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/cluster-locations/<int:user_id>", methods=["GET"])
def trigger_clustering(user_id):
    try:
        cluster_and_save_user_locations(user_id)
        return jsonify({"message": "✅ Zones Updated"}), 200  # 🔧 Matched frontend-friendly response
    except Exception as e:
        print(f"❌ Clustering failed: {str(e)}")
        return jsonify({"error": str(e)}), 500



@app.route("/evaluate-model/<int:user_id>", methods=["GET"])
def evaluate_model(user_id):
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    import numpy as np
    from database.models import AlertHistory

    with app.app_context():
        records = AlertHistory.query.filter_by(user_id=user_id).all()
        if not records:
            return jsonify({"error": "No alert history"}), 400

        # Coordinates as input (X) and labels as output (y)
        X = np.array([[r.latitude, r.longitude] for r in records])
        y_true = np.array([1 if r.location_type == "safe" else 0 for r in records])

        # Train model if needed
        if user_id not in knn_models:
            train_knn_model(user_id)

        model_data = knn_models.get(user_id)
        if not model_data:
            return jsonify({"error": "Model not trained"}), 500
        model = model_data["model"]

        y_pred = model.predict(X)

        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        accuracy = (y_true == y_pred).mean()

        return jsonify({
            "mae": round(mae, 4),
            "mse": round(mse, 4),
            "rmse": round(rmse, 4),
            "accuracy": round(accuracy, 4)
        })


@app.route("/ai/evaluate", methods=["GET"])
def evaluate_ai_model():
    from database.models import UserLocation
    with app.app_context():
        user_id = request.args.get("user_id", type=int)
        if not user_id:
            return jsonify({"error": "Missing user_id"}), 400

        locations = UserLocation.query.filter_by(user_id=user_id, visible=True).all()
        if len(locations) < 4:
            return jsonify({"error": "Not enough data to evaluate"}), 400

        X = [[loc.latitude, loc.longitude] for loc in locations]
        y = [1 if loc.location_type == "safe" else 0 for loc in locations]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        results = {
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
            "f1_score": round(f1_score(y_test, y_pred, zero_division=0), 4)
        }

        return jsonify(results)


@app.route("/retrain-all", methods=["GET"])
def retrain_all_models():
    from database.models import UserLocation
    from sqlalchemy import distinct

    with app.app_context():
        try:
            # Get unique user_ids who have at least 1 location
            user_ids = db.session.query(distinct(UserLocation.user_id)).all()
            user_ids = [uid[0] for uid in user_ids]

            retrained = []
            for user_id in user_ids:
                train_knn_model(user_id, force=True)
                retrained.append(user_id)
                print(f"🔄 Retrained model for user {user_id}")

            return jsonify({
                "message": "✅ Models retrained",
                "users_retrained": retrained
            }), 200

        except Exception as e:
            print(f"❌ Error during bulk retrain: {e}")
            return jsonify({"error": str(e)}), 500


def auto_retrain_loop():
    while True:
        time.sleep(900)  # Wait 1 hour
        with app.app_context():
            print("🔁 [AUTO] Running model retraining for all users...")
            retrain_all_models()
            print(f"✅ [AUTO] Models retrained at {datetime.now(timezone.utc)} UTC")
# Optional: Only start this if enabled
if os.getenv("ENABLE_AUTO_RETRAIN", "false").lower() == "true":
    threading.Thread(target=auto_retrain_loop, daemon=True).start()
    print("🧠 Auto retrain loop started.")


def auto_news_update_loop():
    from sqlalchemy.orm import scoped_session
    while True:
        time.sleep(3600)  # Wait 1 hour
        with app.app_context():
            try:
                print("📰 [AUTO] Running news update...")

                # Replace 'London' with major city or loop through user areas
                city = "London"
                response = app.test_client().get(f"/local-news?area={city}")
                print(f"✅ News refresh for {city}: {response.status_code}")

            except Exception as e:
                print("❌ News fetch loop error:", str(e))



if __name__ == "__main__":
    print("🔥 Flask app is launching...", flush=True)
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)

