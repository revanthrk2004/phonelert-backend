import os
from datetime import timedelta  

class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "supersecretkey")

    
    SQLALCHEMY_DATABASE_URI = os.getenv(
        "DATABASE_URL",
        "postgresql://revanthrk:NYylTjDRlIXvp14G6U4dlZnGEFpm2zaZ@dpg-cv16j6d6l47c73f3ultg-a.oregon-postgres.render.com/phonelert_db"
    )

    
    if SQLALCHEMY_DATABASE_URI.startswith("postgres://"):
        SQLALCHEMY_DATABASE_URI = SQLALCHEMY_DATABASE_URI.replace("postgres://", "postgresql://", 1)

    SQLALCHEMY_TRACK_MODIFICATIONS = False
    JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "superjwtsecret")
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=1000)
