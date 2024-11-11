from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle

# Load the CNN model without compiling
cnn_model =tf.keras.models.load_model(r'C:\Coding Files\my_project\data\cnn_model.keras')
# Initialize Flask app and allow CORS
app = Flask(__name__)
CORS(app)

# Load other models
with open(r'C:\Coding Files\my_project\data\xgboost_model.pkl', 'rb') as f:
    xgboost_model = pickle.load(f)
with open(r'C:\Coding Files\my_project\data\knn_model.pkl', 'rb') as f:
    knn_model = pickle.load(f)
with open(r'C:\Coding Files\my_project\data\model_logistic.pkl', 'rb') as f:
    logistic_model = pickle.load(f)
with open(r'C:\Coding Files\my_project\data\random_forest_model.pkl', 'rb') as f:
    random_forest_model = pickle.load(f)

# Route for rendering the homepage
@app.route('/')
def index():
    return render_template('index.html')

# Route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

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
    app.run(debug=True)
