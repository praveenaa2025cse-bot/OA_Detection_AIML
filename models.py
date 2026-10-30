from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin

db = SQLAlchemy()

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(80), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    pw = db.Column(db.String(256), nullable=False)
    role = db.Column(db.String(20), nullable=False, default="user")

class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    filename = db.Column(db.String(200))
    predicted_class = db.Column(db.String(50))
    confidence = db.Column(db.Float)
    
    # Clinical inputs
    age = db.Column(db.Integer, nullable=True)
    height = db.Column(db.Float, nullable=True)
    weight = db.Column(db.Float, nullable=True)
    bmi = db.Column(db.Float, nullable=True)
    pain_level = db.Column(db.Integer, nullable=True)
    gait_issues = db.Column(db.String(20), nullable=True)
    stiffness = db.Column(db.Boolean, nullable=True)
    notes = db.Column(db.Text, nullable=True)
    
    # Model details
    probabilities = db.Column(db.Text, nullable=True) # JSON string of all class probs
    is_demo = db.Column(db.Boolean, default=False)
    
    created = db.Column(db.DateTime, default=datetime.utcnow)
