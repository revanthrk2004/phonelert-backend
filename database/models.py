from database.db_manager import db
from datetime import datetime



class UserLocation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False)  # User reference
    name = db.Column(db.String(100), nullable=False)  # Location name
    latitude = db.Column(db.Float, nullable=False)
    longitude = db.Column(db.Float, nullable=False)
    radius = db.Column(db.Integer, default=50)  # Radius in meters
    location_type = db.Column(db.String(10), default="unknown")  # ✅ New column (safe, unsafe, unknown)


class PhoneStatus(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), unique=True, nullable=False)  # ✅ Ensure correct FK type
    last_latitude = db.Column(db.Float, nullable=True)
    last_longitude = db.Column(db.Float, nullable=True)
    tracking_active = db.Column(db.Boolean, default=False)  # ✅ Tracks if monitoring is active


class AlertHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False)
    latitude = db.Column(db.Float, nullable=False)
    longitude = db.Column(db.Float, nullable=False)
    location_type = db.Column(db.String(50), nullable=False)  # safe, unsafe, unknown
    ai_decision = db.Column(db.String(20), nullable=False)  # sent, skipped
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)


class User(db.Model):
    __tablename__ = "user"
    __table_args__ = {'extend_existing': True}  # ✅ Allows modification of existing table

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)  # ✅ Add username field
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    linked_devices = db.relationship("LinkedDevice", backref="user", lazy=True)
    saved_locations = db.relationship("SavedLocation", backref="user", lazy=True, cascade="all, delete")

    def set_password(self, password):
        from werkzeug.security import generate_password_hash
        self.password = generate_password_hash(password)  # ✅ Hash password

    def check_password(self, password):
        from werkzeug.security import check_password_hash
        return check_password_hash(self.password, password)  # ✅ Check password

class SavedLocation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    latitude = db.Column(db.Float, nullable=False)
    longitude = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

class LinkedDevice(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    device_name = db.Column(db.String(100), nullable=False, unique=True)  # ✅ Ensure device names are unique
    fcm_token = db.Column(db.String(255), nullable=False, unique=True)  # ✅ Each device should have a unique FCM token


