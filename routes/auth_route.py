from flask import Blueprint, request, jsonify
from flask_jwt_extended import create_access_token, jwt_required, get_jwt_identity
from models.user_model import User
from database.db_manager import db

auth = Blueprint("auth", __name__)

@auth.route("/register", methods=["POST", "OPTIONS"])
def register():
    if request.method == "OPTIONS":
        response = jsonify({})
        response.headers.add("Access-Control-Allow-Origin", "http://localhost:8081")
        response.headers.add("Access-Control-Allow-Methods", "POST, OPTIONS")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type, Authorization")
        response.headers.add("Access-Control-Allow-Credentials", "true")
        return response, 200

    data = request.json
    if User.query.filter_by(email=data["email"]).first():
        return jsonify({"error": "User already exists"}), 400

    user = User(username=data["username"], email=data["email"])
    user.set_password(data["password"])

    db.session.add(user)
    db.session.commit()

    response = jsonify({"message": "User registered successfully!"})
    response.headers.add("Access-Control-Allow-Origin", "http://localhost:8081")
    response.headers.add("Access-Control-Allow-Credentials", "true")
    return response, 201

@auth.route("/login", methods=["POST"])
def login():
    data = request.json
    user = User.query.filter_by(email=data["email"]).first()

    if user and user.check_password(data["password"]):
        access_token = create_access_token(identity=str(user.id))  # Convert ID to string
        return jsonify({"access_token": access_token})

    return jsonify({"error": "Invalid credentials"}), 401

@auth.route("/protected", methods=["GET"])
@jwt_required()
def protected():
    current_user_id = get_jwt_identity()
    return jsonify({"message": "Access granted", "user_id": current_user_id})
