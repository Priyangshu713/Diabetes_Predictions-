
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import numpy as np
import pickle
import os
import tensorflow as tf

# Initialize Flask app and allow CORS
app = Flask(__name__)
CORS(app)

# Define paths
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'data')
CNN_MODEL_PATH = os.path.join(MODEL_DIR, 'cnn_model.keras')
XGBOOST_MODEL_PATH = os.path.join(MODEL_DIR, 'xgboost_model.pkl')
KNN_MODEL_PATH = os.path.join(MODEL_DIR, 'knn_model.pkl')
LOGISTIC_MODEL_PATH = os.path.join(MODEL_DIR, 'model_logistic.pkl')
RANDOM_FOREST_MODEL_PATH = os.path.join(MODEL_DIR, 'random_forest_model.pkl')

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
        data = request.json
        if not data:
            return jsonify({"error": "No input data provided"}), 400
        
        # Assuming a 'features' key in the JSON payload
        features = np.array(data.get('features')).reshape(1, -1)
        predictions = {
            "cnn": cnn_model.predict(features).tolist() if cnn_model else "Error loading CNN model",
            "xgboost": xgboost_model.predict(features).tolist() if xgboost_model else "Error loading XGBoost model",
            "knn": knn_model.predict(features).tolist() if knn_model else "Error loading KNN model",
            "logistic": logistic_model.predict(features).tolist() if logistic_model else "Error loading Logistic Regression model",
            "random_forest": random_forest_model.predict(features).tolist() if random_forest_model else "Error loading Random Forest model"
        }
        return jsonify(predictions)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))  # Use PORT env var, default to 5000 if not set
    app.run(host='0.0.0.0', port=port, debug=True)

