'''
API for Predicting Customer Loyalty Scores
In this code snippet, we define an API endpoint that takes in customer data 
(age, annual income, purchase amount, and purchase frequency) 
and returns a predicted loyalty score. 
The API uses a pre-trained Keras model and a 
preprocessor (scaler) to process the input data and make predictions.

To use this API, you need to have the following files in the same directory:
- predicting_customer_loyalty_preprocessor.joblib: Preprocessor (scaler) for the input data
- loyalty_model.h5: Pre-trained Keras model for predicting loyalty scores

You can run this API:
1. Save and run this code snippet in a file named api_predict_customer_loyalty.py.
2. Run the API using the command python api_predict_customer_loyalty.py.
3. Input this to cmd, make sure to edit the features to match the data you want to predict:
curl -X POST -H "Content-Type: application/json" -d "{\"age\": 35, \"annual_income\": 75000, \"purchase_amount\": 500, \"purchase_frequency\": 6}" http://127.0.0.1:5000/predict_loyalty
4. The API should return a JSON response with the predicted loyalty score.

'''

from tensorflow.keras.losses import mse
from flask import Flask, request, jsonify
import joblib
import tensorflow as tf
import numpy as np

app = Flask(__name__)

preprocessor = joblib.load('predicting_customer_loyalty_preprocessor.joblib')
model = tf.keras.models.load_model('loyalty_model.h5', custom_objects={'mse': mse}) # Pass mse in custom_objects

def predict_loyalty_score(input_features, model, preprocessor):
    input_features_processed = preprocessor.transform(input_features)
    prediction = model.predict(input_features_processed) 
    return prediction.flatten()[0]

@app.route('/predict_loyalty', methods=['POST'])
def api_predict_loyalty():
    try:
        data = request.get_json()

        required_features = ['age', 'annual_income', 'purchase_amount', 'purchase_frequency']
        if not all(feature in data for feature in required_features):
            return jsonify({'error': 'Missing required features in request. Please provide: age, annual_income, purchase_amount, purchase_frequency'}), 400

        input_features = np.array([[
            float(data['age']),
            float(data['annual_income']),
            float(data['purchase_amount']),
            float(data['purchase_frequency'])
        ]])

        predicted_score = predict_loyalty_score(input_features, model, preprocessor)

        return jsonify({'loyalty_score': float(predicted_score)})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)