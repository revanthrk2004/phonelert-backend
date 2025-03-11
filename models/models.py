from database.db_manager import db

class UserLocation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False)  # User reference
    name = db.Column(db.String(100), nullable=False)  # Location name
    latitude = db.Column(db.Float, nullable=False)
    longitude = db.Column(db.Float, nullable=False)
    radius = db.Column(db.Integer, default=50)  # Radius in meters

class PhoneStatus(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False)  # User reference
    last_motion_time = db.Column(db.DateTime, default=datetime.utcnow)  # Last motion timestamp
    last_location = db.Column(db.String(255), nullable=True)  # Last registered location
    is_moving = db.Column(db.Boolean, default=True)  # Motion detected or not

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    last_motion_time = db.Column(db.DateTime, default=datetime.utcnow)
    linked_devices = db.relationship("LinkedDevice", backref="user", lazy=True)

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

db.create_all()  # Create tables if they don’t exist
