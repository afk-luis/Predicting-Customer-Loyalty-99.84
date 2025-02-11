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

'''
THIS IS THE LIGHTWEIGHT VERSION OF THE API
It does not use TensorFlow or joblib.
All calculations are done manually from the preprocessor and model weights.

'''
from flask import Flask, request, Response
import numpy as np

app = Flask(__name__)

# --- Hardcoded Model Parameters ---
# Feature means (averages) for StandardScaler preprocessing
feature_means = np.array([38.6764705882353, 57407.56302521008, 425.6302521008403, 19.798319327731093])
# Feature standard deviations for StandardScaler preprocessing
feature_std_devs = np.array([9.33145224435247, 11379.892776620189, 139.7575250478466, 4.5532882788299975])
# Model weights from the trained linear regression model
model_weights = np.array([0.10322723537683487, 0.34432730078697205, 1.6387391090393066, -0.19711795449256897])
# Model bias from the trained linear regression model
model_bias = np.array([6.786661624908447])

@app.route('/predict_loyalty', methods=['POST'])
def predict_loyalty():
    try:
        data = request.get_json()

        age = float(data.get('age'))
        annual_income = float(data.get('annual_income'))
        purchase_amount = float(data.get('purchase_amount'))
        purchase_frequency = float(data.get('purchase_frequency'))

        features = np.array([[age, annual_income, purchase_amount, purchase_frequency]])

        scaled_features = (features - feature_means) / feature_std_devs

        predicted_score = np.dot(scaled_features, model_weights) + model_bias
        predicted_score = predicted_score.flatten()[0].item()

        if predicted_score > 7:
            value = "High"
            tip = "Make sure to keep this customer happy!"
        elif predicted_score > 4:
            value = "Medium"
            tip = "Consider offering deals to improve loyalty."
        else:
            value = "Low"
            tip = "This customer might need some attention."

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

        return Response(output_text.strip(), mimetype='text/plain')

    except Exception as e:
        error_text = f"Error processing prediction request. Please check input data.\nDetails: {str(e)}"
        return Response(error_text, mimetype='text/plain', status=400)

if __name__ == '__main__':
    app.run(debug=True)
