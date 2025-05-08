from database.db_manager import db

class Location(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    name = db.Column(db.String(50), nullable=False)
    latitude = db.Column(db.Float, nullable=False)
    longitude = db.Column(db.Float, nullable=False)
    visit_count = db.Column(db.Integer, default=1)  
    is_frequent = db.Column(db.Boolean, default=False)  

    def to_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "name": self.name,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "visit_count": self.visit_count,
            "is_frequent": self.is_frequent
        }
