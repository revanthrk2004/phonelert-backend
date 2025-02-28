import sys
import os

# Force Python to recognize 'backend/' as a package
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from flask import Flask, jsonify  # ✅ Import jsonify
from flask_migrate import Migrate  # ✅ Import Flask-Migrate
from database.db_manager import create_app, db
from routes.auth_route import auth
from routes.location_route import location_bp  # Import location routes

app = create_app()

# ✅ Initialize Flask-Migrate
migrate = Migrate(app, db)

# ✅ Add a home route to prevent "Not Found" error
@app.route("/")
def home():
    return jsonify({"message": "Welcome to the Phonelert API! 🎉"}), 200

# Register Blueprints (Routes)
app.register_blueprint(auth, url_prefix="/auth")
app.register_blueprint(location_bp, url_prefix="/location")  # Register location routes

if __name__ == "__main__":
    app.run(debug=True)


