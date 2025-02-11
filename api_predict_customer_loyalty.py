'''
API for Predicting Customer Loyalty Scores (Nicely Formatted Output)
In this code snippet, we define an API endpoint that takes in customer data
(age, annual income, purchase amount, and purchase frequency)
and returns a predicted loyalty score in a user-friendly plain text format.
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
4. The API should return a plain text response with the predicted loyalty score.

'''

from tensorflow.keras.losses import mse
from flask import Flask, request, Response # Changed from jsonify to Response
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
            return Response("Error: Missing required features in request. Please provide: age, annual_income, purchase_amount, purchase_frequency", mimetype='text/plain', status=400) # Plain text error

        age = float(data['age'])
        annual_income = float(data['annual_income'])
        purchase_amount = float(data['purchase_amount'])
        purchase_frequency = float(data['purchase_frequency'])

        input_features = np.array([[
            age,
            annual_income,
            purchase_amount,
            purchase_frequency
        ]])

        predicted_score = predict_loyalty_score(input_features, model, preprocessor)
        if predicted_score > 7:
            value = "High"
            tip = "Make sure to keep this customer happy!"
        elif predicted_score > 4:
            value = "Medium"
            tip = "Consider offering deals to improve loyalty."
        else:
            value = "Low"
            tip = "This customer might need some attention."

        # Construct formatted plain text output
        output_text = f"""
------------------------------------------------------------
                 Customer Loyalty Prediction
                      Customer Details
                        Age:  {age} years old
              Annual Income:  {annual_income:,.2f} USD per year
            Purchase Amount:  {purchase_amount:,.2f} USD per purchase
         Purchase Frequency:  {purchase_frequency} times per year
                              ----------------
    Predicted Loyalty Score:  | {predicted_score:.2f} / 10.00 |
                              ----------------
    This is a {value}-Loyalty Customer.
    {tip}
------------------------------------------------------------
"""
        return Response(output_text.strip(), mimetype='text/plain') # Plain text response

    except Exception as e:
        error_text = f"Error processing prediction request. Please check input data.\nDetails: {str(e)}"
        return Response(error_text, mimetype='text/plain', status=400) # Plain text error

if __name__ == '__main__':
    app.run(debug=True)
