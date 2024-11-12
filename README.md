
# Diabetes Prediction Application

This project is a **Diabetes Prediction Web Application** built using Flask. It provides an interactive web interface where users can input health metrics and receive a prediction on their likelihood of having diabetes. The application integrates multiple machine learning models, including CNN, KNN, logistic regression, and possibly more, to perform predictions based on user input.

## Features

- **Multi-Model Prediction**: Utilizes multiple models to provide predictions, including CNN, logistic regression, and KNN.
- **Interactive Web Interface**: A user-friendly form for inputting data and receiving instant predictions.
- **Model Persistence**: Pre-trained models are saved and loaded on request, avoiding retraining time.
- **Responsive Design**: Works across different screen sizes and devices.

## File Structure

- `app.py`: Main application script initializing Flask and setting up model loading and prediction functions.
- `data/`: Contains pre-trained models and the dataset used for training:
  - `cnn_model.keras`, `cnn_model.h5`: CNN models for prediction.
  - `knn_model.pkl`, `model_logistic.pkl`: KNN and logistic regression models, respectively.
  - `diabetes_prediction_dataset.csv`: Dataset file, likely used for model training or reference.
- `requirements.txt`: Lists dependencies necessary to run the application.
- `static/`: Contains frontend assets, including CSS for styling (`style.css`), JavaScript for client-side interactions (`script.js`), and a favicon (`favicon.ico`).
- `templates/`: Contains the main HTML page (`index.html`) that users interact with.

## Installation

To set up and run the application locally, follow these steps:

1. **Clone the Repository**:
    ```bash
    git clone <repo_url>
    cd Diabetes_Predictions--main
    ```

2. **Install Dependencies**:
   Use `pip` to install the required packages from `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the Application**:
    ```bash
    python app.py
    ```
    The application should now be accessible at `http://127.0.0.1:5000`.

## Usage

1. Navigate to the application's main page.
2. Input the following health metrics:
   - **Age**: Accepts decimal values.
   - **Hypertension**: Binary (0 for no, 1 for yes).
   - **Heart Disease**: Binary (0 for no, 1 for yes).
   - **BMI**: Accepts decimal values.
   - **HbA1c Level**: Accepts decimal values.
   - **Blood Glucose Level**: Accepts decimal values.
3. Click "Submit" to see the diabetes prediction based on the provided data.

## Dependencies

- **Flask 2.3.2**: Web framework to create the backend server.
- **Flask-CORS 5.0.0**: Middleware for enabling Cross-Origin Resource Sharing.
- **Numpy 1.23.5** and **Pandas 2.0.2**: Used for data manipulation.
- **TensorFlow 2.17.0** and **Keras 3.4.1**: For loading and using the CNN model.
- **Gunicorn**: For deploying the application in production environments.

## Models Used

The project includes several models stored in the `data/` folder:
- **CNN Model**: Used for image or pattern recognition on diabetes-related data.
- **KNN Model**: A simple, instance-based learning algorithm.
- **Logistic Regression Model**: Linear model for binary classification.
- **XGBoost Model**: A powerful, gradient-boosted tree model well-suited for tabular data.

## Features
Expanded Model Selection: In addition to CNN, logistic regression, and KNN models, the application also includes XGBoost models to improve prediction accuracy and provide a robust comparison of algorithms.

## Future Improvements

- Extend the application to support additional health metrics.
- Implement a feedback loop for model retraining based on new user data.
- Add a comprehensive data visualization section.

## License

This project is licensed under the MIT License.See the [LICENSE](LICENSE) file for more information.
