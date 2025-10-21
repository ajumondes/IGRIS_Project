# app.py

import os
import json
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

import lstm_model # Import our new model

# --- App Initialization & Configuration ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'a-very-secret-key-that-you-should-change'
basedir = os.path.abspath(os.path.dirname(__file__))
instance_path = os.path.join(basedir, 'instance')
os.makedirs(instance_path, exist_ok=True)
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{os.path.join(instance_path, "igris.db")}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# --- Database & Login Manager Setup ---
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# --- Database Models ---
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    def set_password(self, password): self.password_hash = generate_password_hash(password)
    def check_password(self, password): return check_password_hash(self.password_hash, password)

class UserProfile(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    model_path = db.Column(db.String(255), nullable=True) 
    scaler_path = db.Column(db.String(255), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class AuthLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    trust_score = db.Column(db.Float, nullable=False)
    decision = db.Column(db.String(50), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# --- Web Page Routes ---
@app.route('/')
def home(): return render_template('home.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated: return redirect(url_for('home'))
    if request.method == 'POST':
        user_exists = User.query.filter_by(username=request.form.get('username')).first()
        if user_exists: flash('Username already exists.', 'danger'); return redirect(url_for('register'))
        new_user = User(username=request.form.get('username')); new_user.set_password(request.form.get('password'))
        db.session.add(new_user); db.session.commit()
        flash('Registration successful! Please log in.', 'success'); return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated: return redirect(url_for('home'))
    if request.method == 'POST':
        user = User.query.filter_by(username=request.form.get('username')).first()
        if user and user.check_password(request.form.get('password')):
            login_user(user); return redirect(url_for('home'))
        else: flash('Invalid username or password.', 'danger')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout(): logout_user(); return redirect(url_for('login'))

@app.route('/enroll')
@login_required
def enroll(): return render_template('enroll.html')

# --- API Endpoints ---
@app.route('/api/enroll', methods=['POST'])
@login_required
def api_enroll():
    raw_events = request.get_json()
    if not raw_events:
        return jsonify({'status': 'error', 'message': 'No data received.'}), 400
    
    print("Starting model training... This may take a moment.")
    model_path, scaler_path = lstm_model.train_user_model(current_user.id, raw_events)
    
    if not model_path:
        return jsonify({'status': 'error', 'message': 'Not enough data to create a profile.'}), 400
    
    profile = UserProfile.query.filter_by(user_id=current_user.id).first()
    if profile:
        profile.model_path = model_path
        profile.scaler_path = scaler_path
    else:
        profile = UserProfile(user_id=current_user.id, model_path=model_path, scaler_path=scaler_path)
        db.session.add(profile)
    
    db.session.commit()
    return jsonify({'status': 'success', 'message': 'LSTM profile created successfully!'})

@app.route('/api/authenticate', methods=['POST'])
def api_authenticate():
    data = request.get_json()
    if not data: return jsonify({'trust_score': 0.0, 'decision': 'No data received'})
    username = data.get('user_id')
    keyboard_events = data.get('keyboard_events', [])
    user = User.query.filter_by(username=username).first()
    if not user: return jsonify({'trust_score': 0.0, 'decision': f'User {username} not found'})
    if not keyboard_events: return jsonify({'trust_score': 0.0, 'decision': 'No keyboard data'})
    profile = UserProfile.query.filter_by(user_id=user.id).first()
    if not profile or not profile.model_path: return jsonify({'trust_score': 0.0, 'decision': 'No profile for user'})

    trust_score = lstm_model.get_model_score(profile, keyboard_events)
    decision = "Genuine" if trust_score > 0.5 else "Anomaly" 
    
    db.session.add(AuthLog(user_id=user.id, trust_score=trust_score, decision=decision))
    db.session.commit()
    return jsonify({'trust_score': trust_score, 'decision': decision})

# --- Main Execution ---
if __name__ == '__main__':
    with app.app_context():
        db.create_all() # This creates the .db file and tables
    app.run(debug=True)