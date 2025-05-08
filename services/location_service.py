from models.location_model import Location
from database.db_manager import db


FREQUENCY_THRESHOLD = 5

def update_location_frequency(user_id, location_name):
    location = Location.query.filter_by(user_id=user_id, name=location_name).first()
    
    if location:
        location.visit_count += 1

        
        if location.visit_count >= FREQUENCY_THRESHOLD:
            location.is_frequent = True

        db.session.commit()
