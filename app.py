from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_mail import Mail, Message
from itsdangerous import URLSafeTimedSerializer, SignatureExpired
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure app
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'default_secret_key')  # Use env var for prod

# Mail configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME', 'your_email@gmail.com')  # Replace with your email
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD', 'your_password')        # Replace with your password
app.config['MAIL_DEFAULT_SENDER'] = os.getenv('MAIL_DEFAULT_SENDER', 'your_email@gmail.com')  # Replace with your email
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
mail = Mail(app)
s = URLSafeTimedSerializer(app.config['MAIL_USERNAME'])

# User model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)

    def __repr__(self):
        return f"User('{self.username}', '{self.email}', Verified: {self.is_verified})"

# Function to create dynamic user-specific tables
def create_user_table(username):
    class DynamicUserTable(db.Model):
        __tablename__ = f"user_{username}"
        id = db.Column(db.Integer, primary_key=True)
        entry = db.Column(db.String(200), nullable=False)
        
        def __repr__(self):
            return f"{self.__tablename__} Entry('{self.entry}')"

    db.create_all()  # Ensures the new table is created in the database
    return DynamicUserTable

from flask import current_app  # For logging

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        try:
            # Log incoming request data for debugging
            app.logger.info(f"Request Form Data: {request.form}")

            # Validate incoming form data
            username = request.form.get('username')
            email = request.form.get('email')
            password = request.form.get('password')

            # Check if all fields are provided
            if not username or not email or not password:
                flash('All fields are required: username, email, and password.', 'danger')
                app.logger.warning("Missing fields in registration form.")
                return jsonify({'error': 'All fields are required'}), 400

            # Hash the password
            hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')

            # Check if the user already exists
            if User.query.filter_by(username=username).first():
                flash('Username already exists. Please choose a different one.', 'danger')
                return jsonify({'error': 'Username already exists'}), 400

            if User.query.filter_by(email=email).first():
                flash('Email already registered. Please use a different email.', 'danger')
                return jsonify({'error': 'Email already registered'}), 400

            # Create and save the new user
            user = User(username=username, email=email, password=hashed_password)
            db.session.add(user)
            db.session.commit()

            # Generate and send verification email
            token = s.dumps(email, salt='email-confirm')
            link = url_for('confirm_email', token=token, _external=True)
            msg = Message('Confirm Your Email', sender=app.config['MAIL_USERNAME'], recipients=[email])
            msg.body = f'Click the link to verify your email: {link}'
            mail.send(msg)

            flash('A verification email has been sent. Please check your inbox.', 'info')
            return jsonify({'success': 'User registered successfully. Verification email sent.'}), 200

        except Exception as e:
            # Log error and return a friendly response
            app.logger.error(f"Error during registration: {str(e)}")
            flash('An error occurred during registration. Please try again.', 'danger')
            return jsonify({'error': 'An error occurred during registration. Please try again later.'}), 500

    # For GET requests, return the registration page
    return render_template('register.html')



@app.route('/confirm_email/<token>')
def confirm_email(token):
    try:
        email = s.loads(token, salt='email-confirm', max_age=3600)
        user = User.query.filter_by(email=email).first()
        if user:
            user.is_verified = True
            db.session.commit()
            flash('Email verified successfully!', 'success')
            return redirect(url_for('login'))
    except SignatureExpired:
        flash('The confirmation link has expired.', 'danger')
        return redirect(url_for('register'))
    return redirect(url_for('login'))

# Password reset
@app.route('/reset_request', methods=['GET', 'POST'])
def reset_request():
    if request.method == 'POST':
        email = request.form['email']
        user = User.query.filter_by(email=email).first()
        if user:
            token = s.dumps(email, salt='password-reset')
            link = url_for('reset_password', token=token, _external=True)

            msg = Message('Password Reset Request', sender=app.config['MAIL_USERNAME'], recipients=[email])
            msg.body = f'Click the link to reset your password: {link}'
            mail.send(msg)
            flash('A password reset link has been sent to your email.', 'info')
        else:
            flash('Email not found.', 'danger')
    return render_template('reset_request.html')

@app.route('/reset_password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    try:
        email = s.loads(token, salt='password-reset', max_age=3600)
        if request.method == 'POST':
            user = User.query.filter_by(email=email).first()
            user.password = bcrypt.generate_password_hash(request.form['password']).decode('utf-8')
            db.session.commit()
            flash('Your password has been reset!', 'success')
            return redirect(url_for('login'))
    except SignatureExpired:
        flash('The reset link has expired.', 'danger')
    return render_template('reset_password.html')

# Load models
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'data')
MODEL_PATHS = {
    'cnn': os.path.join(MODEL_DIR, 'cnn_model.keras'),
    'xgboost': os.path.join(MODEL_DIR, 'xgboost_model.pkl'),
    'knn': os.path.join(MODEL_DIR, 'knn_model.pkl'),
    'logistic': os.path.join(MODEL_DIR, 'model_logistic.pkl'),
    'random_forest': os.path.join(MODEL_DIR, 'random_forest_model.pkl')
}

def load_model(model_path, model_type='pickle'):
    if model_type == 'keras':
        return tf.keras.models.load_model(model_path)
    with open(model_path, 'rb') as f:
        return pickle.load(f)

models = {
    'cnn': load_model(MODEL_PATHS['cnn'], model_type='keras'),
    'xgboost': load_model(MODEL_PATHS['xgboost']),
    'knn': load_model(MODEL_PATHS['knn']),
    'logistic': load_model(MODEL_PATHS['logistic']),
    'random_forest': load_model(MODEL_PATHS['random_forest'])
}

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = np.array([[data['age'], data['hypertension'], data['heart_disease'], data['bmi'], data['HbA1c_level'], data['blood_glucose_level']]])
        features_df = pd.DataFrame(features, columns=['age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level'])

        predictions = {
            'cnn': 'Diabetic' if models['cnn'].predict(features.reshape(1, -1, 1))[0][0] >= 0.5 else 'Not Diabetic',
            'xgboost': 'Diabetic' if models['xgboost'].predict(features_df)[0] == 1 else 'Not Diabetic',
            'knn': 'Diabetic' if models['knn'].predict(features_df)[0] == 1 else 'Not Diabetic',
            'logistic': 'Diabetic' if models['logistic'].predict(features_df)[0] == 1 else 'Not Diabetic',
            'random_forest': 'Diabetic' if models['random_forest'].predict(features_df)[0] == 1 else 'Not Diabetic'
        }

        return jsonify({'success': True, 'predictions': predictions})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Main block
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)), debug=True)
