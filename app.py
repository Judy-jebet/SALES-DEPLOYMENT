from flask import Flask, render_template, request, jsonify
import joblib  
import pickle
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(filename='app.log', level=logging.INFO)

# Load your pre-trained model
try:
    with open('linear_regression_model(1).pkl', 'rb') as file:
        model = pickle.load(file)
        logging.info("Model loaded successfully")
except FileNotFoundError:
    logging.error("Model file not found")

# Define your prediction function using the loaded model
def predict_sales(product, branch, city, payment, customerType):
    try:
        # Make prediction using the loaded model
        prediction = model.predict([[product, branch, city, payment, customerType]])  # Adjust as per your model input
        return prediction[0]  # Return the predicted value
    except Exception as e:
        logging.error(f"Prediction failed: {str(e)}")
        return None

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        product = request.form['product']
        branch = request.form['branch']
        city = request.form['city']
        payment = request.form['payment']
        customerType = request.form['customerType']

        # Make prediction
        predicted_sales = predict_sales(product, branch, city, payment, customerType)
        
        if predicted_sales is not None:
            return jsonify({'predicted_sales': predicted_sales})  # Return prediction as JSON response
        else:
            return jsonify({'error': 'Prediction failed. Please check server logs for details.'}), 500

    return render_template('index.html')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=9090, debug=True)
