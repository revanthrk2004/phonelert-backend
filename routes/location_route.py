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
        location.visit_count += 1  
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

    if "name" not in data or "latitude" not in data or "longitude" not in data:
        return jsonify({"error": "Missing required fields (name, latitude, longitude)!"}), 400

    location = Location.query.filter_by(user_id=user_id, name=data["name"]).first()

    if location:
        update_location_frequency(user_id, data["name"])
        return jsonify({"message": "Location visit count updated!", "location": location.to_dict()}), 200

    
    new_location = Location(
        user_id=user_id,
        name=data["name"],
        latitude=data["latitude"],
        longitude=data["longitude"],
        visit_count=1,  
        is_frequent=False  
    )

    db.session.add(new_location)
    db.session.commit()

    return jsonify({"message": "Location added successfully!", "location": new_location.to_dict()}), 201

@location_bp.route("/get_frequent_locations", methods=["GET"])
@jwt_required()
def get_frequent_locations():
    """
    Fetch all locations that have been marked as frequent.
    """
    user_id = get_jwt_identity()
    locations = Location.query.filter_by(user_id=user_id, is_frequent=True).all()

    if not locations:
        return jsonify({"message": "No frequent locations found!"}), 404

    return jsonify([location.to_dict() for location in locations]), 200

@location_bp.route("/get_all_locations", methods=["GET"])
@jwt_required()
def get_all_locations():
    """
    Fetch all locations belonging to the authenticated user.
    """
    user_id = get_jwt_identity()
    locations = Location.query.filter_by(user_id=user_id).all()

    if not locations:
        return jsonify({"message": "No locations found!"}), 404

    return jsonify([location.to_dict() for location in locations]), 200

@location_bp.route("/get_location/<int:location_id>", methods=["GET"])
@jwt_required()
def get_location_by_id(location_id):
    """
    Fetch a single location by its ID.
    """
    user_id = get_jwt_identity()
    location = Location.query.filter_by(id=location_id, user_id=user_id).first()

    if not location:
        return jsonify({"error": "Location not found!"}), 404

    return jsonify(location.to_dict()), 200

@location_bp.route("/delete/<int:location_id>", methods=["DELETE"])
@jwt_required()
def delete_location(location_id):
    """
    Delete a location if it belongs to the authenticated user.
    """
    user_id = get_jwt_identity()
    location = Location.query.filter_by(id=location_id, user_id=user_id).first()

    if not location:
        return jsonify({"error": "Location not found or does not belong to the user"}), 404

    db.session.delete(location)
    db.session.commit()

    return jsonify({"message": "Location deleted successfully!"}), 200

@location_bp.route("/update/<int:location_id>", methods=["PUT"])
@jwt_required()
def update_location_name(location_id):
    """
    Update the name of an existing location.
    """
    user_id = get_jwt_identity()
    data = request.json

    if "new_name" not in data or not data["new_name"].strip():
        return jsonify({"error": "New name cannot be empty!"}), 400

    location = Location.query.filter_by(id=location_id, user_id=user_id).first()

    if not location:
        return jsonify({"error": "Location not found!"}), 404

    location.name = data["new_name"].strip()
    db.session.commit()

    return jsonify({"message": "Location name updated successfully!", "location": location.to_dict()}), 200




