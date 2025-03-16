from database.db_manager import db
from datetime import datetime


class UserLocation(db.Model):
    location_name = db.Column(db.String(100), primary_key=True)  # ✅ Now the primary key
    latitude = db.Column(db.Float, nullable=False)
    longitude = db.Column(db.Float, nullable=False)
    location_type = db.Column(db.String(10), default="safe")  # ✅ "safe" or "unsafe" (chosen by user)
    radius = db.Column(db.Integer, default=50)  # ✅ Default radius (meters)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)  # ✅ Track when the location was added

    def __repr__(self):
        return f"<UserLocation {self.location_name}, Type={self.location_type}>"



class PhoneStatus(db.Model):
    location_name = db.Column(db.String(100), primary_key=True)  # ✅ Now the primary key
    last_latitude = db.Column(db.Float, nullable=True)
    last_longitude = db.Column(db.Float, nullable=True)
    tracking_active = db.Column(db.Boolean, default=False)  # ✅ Tracks if monitoring is active

    def __repr__(self):
        return f"<PhoneStatus {self.location_name}, Tracking={self.tracking_active}>"


 
class User(db.Model):
    __tablename__ = "user"
    __table_args__ = {'extend_existing': True}  # ✅ FIXED: Allows modification of existing table

    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    linked_devices = db.relationship("LinkedDevice", backref="user", lazy=True)
    saved_locations = db.relationship("SavedLocation", backref="user", lazy=True, cascade="all, delete")

class SavedLocation(db.Model):
    location_name = db.Column(db.String(100), primary_key=True)  # ✅ Now the primary key
    latitude = db.Column(db.Float, nullable=False)
    longitude = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<SavedLocation {self.location_name}>"


class LinkedDevice(db.Model):
    location_name = db.Column(db.String(100), primary_key=True)  # ✅ Now the primary key
    device_name = db.Column(db.String(100), nullable=False, unique=True)  # ✅ Ensure device names are unique
    fcm_token = db.Column(db.String(255), nullable=False, unique=True)  # ✅ Each device should have a unique FCM token

    def __repr__(self):
        return f"<LinkedDevice {self.device_name} for {self.location_name}>"


class AlertHistory(db.Model):
    __tablename__ = "alert_history"

    location_name = db.Column(db.String(100), primary_key=True)  # ✅ Now the primary key
    latitude = db.Column(db.Float, nullable=False)
    longitude = db.Column(db.Float, nullable=False)
    location_type = db.Column(db.String(20), nullable=False)  # safe or unsafe (user-defined)
    ai_decision = db.Column(db.String(10), nullable=False)  # sent or skipped
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<AlertHistory Location={self.location_name}, AI Decision={self.ai_decision}>"


    def __repr__(self):
        return f"<AlertHistory user_id={self.user_id}, location_type={self.location_type}, ai_decision={self.ai_decision}>"
