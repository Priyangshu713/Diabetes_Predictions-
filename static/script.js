function predict() {
    // Collect form data
    const data = {
        age: document.getElementById('age').value,
        hypertension: document.getElementById('hypertension').value,
        heart_disease: document.getElementById('heart_disease').value,
        bmi: document.getElementById('bmi').value,
        HbA1c_level: document.getElementById('HbA1c_level').value,
        blood_glucose_level: document.getElementById('blood_glucose_level').value
    };

    // Send data to backend
    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(result => {
        // Display prediction
        const resultDiv = document.getElementById('result');
        resultDiv.innerHTML = '<h2>Prediction Results:</h2>'; 

        // Iterate over the result and display each model's prediction
        for (const model in result.prediction) { // Access the 'prediction' property of the result
            resultDiv.innerHTML += `<p><strong>${model}</strong>: ${result.prediction[model]}</p>`; // Access the value for each model
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('An error occurred while making the prediction.');
    });
}
