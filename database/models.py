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
     
    visible = db.Column(db.Boolean, default=True)
    
    def __repr__(self):
        return f"<UserLocation {self.location_name}, User={self.user_id}, Type={self.location_type}>"



class PhoneStatus(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), unique=True, nullable=False)  
    last_latitude = db.Column(db.Float, nullable=True)
    last_longitude = db.Column(db.Float, nullable=True)
    tracking_active = db.Column(db.Boolean, default=False)  
    expo_push_token = db.Column(db.String, nullable=True) 


 
from models.user_model import User  

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
    was_anomaly = db.Column(db.Boolean, default=False)
    reason = db.Column(db.String(100))
    risk_score = db.Column(db.Float, nullable=True)  

    def __repr__(self):
        return f"<AlertHistory user_id={self.user_id}, location_type={self.location_type}, ai_decision={self.ai_decision}, risk_score={self.risk_score}>"


class LocationVisitLog(db.Model):
    __tablename__ = "location_visit_log"
    user_id = db.Column(db.Integer, primary_key=True)
    latitude = db.Column(db.Float, primary_key=True)
    longitude = db.Column(db.Float, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<VisitLog user={self.user_id} at ({self.latitude}, {self.longitude})>"
