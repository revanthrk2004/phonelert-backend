from flask import Blueprint, request, jsonify
from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity
from database.models import User
from database.db_manager import db
from werkzeug.security import check_password_hash
import datetime
import os
from flask import request




auth = Blueprint("auth", __name__)

@auth.route("/register", methods=["POST"])
def register():
    data = request.json
    if User.query.filter_by(email=data["email"]).first():
        return jsonify({"error": "User already exists"}), 400

    user = User(username=data["username"], email=data["email"])
    user.set_password(data["password"])
    
    db.session.add(user)
    db.session.commit()
    
    return jsonify({"message": "User registered successfully!", "user_id": user.id}), 201  # ✅ Include user_id

from flask import request

@auth.route("/login", methods=["POST"])
def login():
    if not request.is_json:
        return jsonify({"error": "Invalid JSON format"}), 400  # ✅ Handle non-JSON request

    data = request.get_json()
    
    if not data or "email" not in data or "password" not in data:
        return jsonify({"error": "Missing email or password"}), 400

    user = User.query.filter_by(email=data["email"]).first()

    if user and user.check_password(data["password"]):
        access_token = create_access_token(identity=str(user.id))
        return jsonify({"access_token": access_token, "user_id": user.id, "username": user.username})

    return jsonify({"error": "Invalid credentials"}), 401

@auth.route("/protected", methods=["GET"])
@jwt_required()
def protected():
    current_user_id = get_jwt_identity()
    return jsonify({"message": "Access granted", "user_id": current_user_id})
