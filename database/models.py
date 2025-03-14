from database.db_manager import db
from datetime import datetime


class UserLocation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False)  # User reference
    name = db.Column(db.String(100), nullable=False)  # Location name
    latitude = db.Column(db.Float, nullable=False)
    longitude = db.Column(db.Float, nullable=False)
    radius = db.Column(db.Integer, default=50)  # Radius in meters

class PhoneStatus(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(100), unique=True, nullable=False)
    last_latitude = db.Column(db.Float, nullable=True)
    last_longitude = db.Column(db.Float, nullable=True)
    tracking_active = db.Column(db.Boolean, default=False)  # ✅ NEW FLAG

 
class User(db.Model):
    __tablename__ = "user"
    __table_args__ = {'extend_existing': True}  # ✅ FIXED: Allows modification of existing table

    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    linked_devices = db.relationship("LinkedDevice", backref="user", lazy=True)
    saved_locations = db.relationship("SavedLocation", backref="user", lazy=True, cascade="all, delete")

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
    device_name = db.Column(db.String(100), nullable=False)
    fcm_token = db.Column(db.String(255), nullable=False)  # Firebase Token for Notifications

