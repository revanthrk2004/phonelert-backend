from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from models.location_model import Location
from database.db_manager import db

location_bp = Blueprint("location", __name__)

def update_location_frequency(user_id, location_name):
    """
    Update the visit count of a location and mark it as frequent if visited 3+ times.
    """
    location = Location.query.filter_by(user_id=user_id, name=location_name).first()

    if location:
        location.visit_count += 1  # Increase visit count

        # 🔥 If location is visited 3+ times, mark it as frequent
        if location.visit_count >= 3:
            location.is_frequent = True

        db.session.commit()

@location_bp.route("/add_location", methods=["POST"])
@jwt_required()
def add_location():
    """
    Add a new location or update the visit count if it already exists.
    """
    data = request.json
    user_id = get_jwt_identity()

    location = Location.query.filter_by(user_id=user_id, name=data["name"]).first()

    if location:
        update_location_frequency(user_id, data["name"])
        return jsonify({"message": "Location visit count updated!"}), 200

    # Create new location
    new_location = Location(
        user_id=user_id,
        name=data["name"],
        latitude=data["latitude"],
        longitude=data["longitude"],
        visit_count=1,  # Start with 1 visit
        is_frequent=False  # Not frequent yet
    )

    db.session.add(new_location)
    db.session.commit()

    return jsonify({"message": "Location added successfully!"}), 201

@location_bp.route("/get_frequent_locations", methods=["GET"])
@jwt_required()
def get_frequent_locations():
    """
    Fetch all locations that have been marked as frequent.
    """
    user_id = get_jwt_identity()
    locations = Location.query.filter_by(user_id=user_id, is_frequent=True).all()

    return jsonify([location.to_dict() for location in locations]), 200

