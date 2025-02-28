import os
from datetime import timedelta  # <-- Ensure this is imported

class Config:
    SECRET_KEY = "supersecretkey"
    SQLALCHEMY_DATABASE_URI = "sqlite:///phonelert.db"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    JWT_SECRET_KEY = "superjwtsecret"
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=2)  # <-- Indent this correctly inside the class

