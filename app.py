from flask import Flask, render_template, request, jsonify
import joblib  # Use joblib to load scikit-learn models, or import your model loading library
import pickle

app = Flask(__name__)

# Load your pre-trained model
model = joblib.load('/workspaces/SALES-DEPLOYMENT/model/linear_regression_model.pkl')  # Adjust the filename as per your model file

# Define your prediction function using the loaded model
def predict_sales(product, branch, city, payment, customerType):
    # Your preprocessing steps here if needed
    # Example: Convert categorical variables to numerical using one-hot encoding

    # Make prediction using the loaded model
    prediction = model.predict([[product, branch, city, payment, customerType]])  # Adjust as per your model input
    return prediction[0]  # Return the predicted value

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

        return jsonify({'predicted_sales': predicted_sales})  # Return prediction as JSON response

    return render_template('index.html')

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8090, debug=True)

