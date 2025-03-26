from database.db_manager import db
from datetime import datetime


class UserLocation(db.Model):
    __tablename__ = 'user_location'

    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), primary_key=True)
    latitude = db.Column(db.Float, primary_key=True)
    longitude = db.Column(db.Float, primary_key=True)
    location_name = db.Column(db.String(100))
    location_type = db.Column(db.String(10), default="safe")
    radius = db.Column(db.Integer, default=50)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
     # 👇 NEW LINE
    visible = db.Column(db.Boolean, default=True)
    
    def __repr__(self):
        return f"<UserLocation {self.location_name}, User={self.user_id}, Type={self.location_type}>"



class PhoneStatus(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), unique=True, nullable=False)  # ✅ Ensure correct FK type
    last_latitude = db.Column(db.Float, nullable=True)
    last_longitude = db.Column(db.Float, nullable=True)
    tracking_active = db.Column(db.Boolean, default=False)  # ✅ Tracks if monitoring is active


 
from models.user_model import User  # import instead of redefining

class SavedLocation(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    latitude = db.Column(db.Float, nullable=False)
    longitude = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)


class AlertHistory(db.Model):
    __tablename__ = "alert_history"

    user_id = db.Column(db.Integer, primary_key=True)
    latitude = db.Column(db.Float, primary_key=True)
    longitude = db.Column(db.Float, primary_key=True)
    location_type = db.Column(db.String(20), nullable=False)
    ai_decision = db.Column(db.String(10), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    # ✅ New fields for anomaly detection
    was_anomaly = db.Column(db.Boolean, default=False)  # ← If AI flagged it as anomaly
    reason = db.Column(db.String(200), nullable=True)   # ← Why it was flagged

    def __repr__(self):
        return f"<AlertHistory user_id={self.user_id}, location_type={self.location_type}, ai_decision={self.ai_decision}, anomaly={self.was_anomaly}>"


class LocationVisitLog(db.Model):
    __tablename__ = "location_visit_log"
    user_id = db.Column(db.Integer, primary_key=True)
    latitude = db.Column(db.Float, primary_key=True)
    longitude = db.Column(db.Float, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<VisitLog user={self.user_id} at ({self.latitude}, {self.longitude})>"
