from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import os


# Initialize Flask app and allow CORS
app = Flask(__name__)
CORS(app)

# Load other models
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'data')
CNN_MODEL_PATH = os.path.join(MODEL_DIR, 'cnn_model.keras')
XGBOOST_MODEL_PATH = os.path.join(MODEL_DIR, 'xgboost_model.pkl')
KNN_MODEL_PATH = os.path.join(MODEL_DIR, 'knn_model.pkl')
LOGISTIC_MODEL_PATH = os.path.join(MODEL_DIR, 'model_logistic.pkl')
RANDOM_FOREST_MODEL_PATH = os.path.join(MODEL_DIR, 'random_forest_model.pkl')


# Print model paths for debugging 
# print(f"Looking for CNN model at: {CNN_MODEL_PATH}")
# print(f"Looking for XGBoost model at: {XGBOOST_MODEL_PATH}")
# print(f"Looking for KNN model at: {KNN_MODEL_PATH}")
# print(f"Looking for Logistic model at: {LOGISTIC_MODEL_PATH}")
# print(f"Looking for Random Forest model at: {RANDOM_FOREST_MODEL_PATH}")

# Load models with error handling
def load_model(model_path, model_type='pickle'):
    try:
        if model_type == 'keras':
            return tf.keras.models.load_model(model_path)
        else:
            with open(model_path, 'rb') as f:
                return pickle.load(f)
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return None

cnn_model = load_model(CNN_MODEL_PATH, model_type='keras')
xgboost_model = load_model(XGBOOST_MODEL_PATH)
knn_model = load_model(KNN_MODEL_PATH)
logistic_model = load_model(LOGISTIC_MODEL_PATH)
random_forest_model = load_model(RANDOM_FOREST_MODEL_PATH)


# Route for rendering the homepage
@app.route('/')
def index():
    return render_template('index.html')

# Route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "No input data provided"}), 400
        
        # Extract features from the request
        age = float(data['age'])
        hypertension = int(data['hypertension'])
        heart_disease = int(data['heart_disease'])
        bmi = float(data['bmi'])
        HbA1c_level = float(data['HbA1c_level'])
        glucose = float(data['blood_glucose_level'])

        # Prepare features array
        features = np.array([[age, hypertension, heart_disease, bmi, HbA1c_level, glucose]])

        # Convert numpy array to pandas DataFrame with appropriate feature names
        features_df = pd.DataFrame(features, columns=['age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level'])

        # Predict with each model
        cnn_pred = cnn_model.predict(features.reshape(1, -1, 1))[0][0]
        xgboost_pred = xgboost_model.predict(features_df)[0]
        knn_pred = knn_model.predict(features_df)[0]
        logistic_pred = logistic_model.predict(features_df)[0]
        random_forest_pred = random_forest_model.predict(features_df)[0]

        #convert numpy array to float and int
        cnn_pred = float(cnn_pred)
        xgboost_pred = int(xgboost_pred)
        knn_pred = int(knn_pred)
        logistic_pred = int(logistic_pred)
        random_forest_pred = int(random_forest_pred)

        # Prepare the result, with 'xgboost' at the top
        result = {
            'xgboost': 'Diabetic' if xgboost_pred == 1 else 'Not Diabetic',
            'cnn': 'Diabetic' if cnn_pred > 0.5 else 'Not Diabetic',
            'knn': 'Diabetic' if knn_pred == 1 else 'Not Diabetic',
            'logistic': 'Diabetic' if logistic_pred == 1 else 'Not Diabetic',
            'random_forest': 'Diabetic' if random_forest_pred == 1 else 'Not Diabetic'
        }

        return jsonify({'success': True, 'prediction': result})
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': f'An error occurred while making the prediction: {str(e)}'}), 500


# Main block to run the Flask app
if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))  # Use PORT env var, default to 5000 if not set
    app.run(host='0.0.0.0', port=port, debug=True)
