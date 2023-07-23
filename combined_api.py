from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pickle

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained KNN models for each nutrient
models = {
    'Nitrogen': joblib.load('model_N.joblib'),
    'Phosphorous': joblib.load('model_P.joblib'),
    'Potassium': joblib.load('model_K.joblib'),
    'Calcium': joblib.load('model_Ca.joblib'),
    'Magnesium': joblib.load('model_Mg.joblib'),
    'Sulphur': joblib.load('model_S.joblib'),
    'Iron': joblib.load('model_Fe.joblib'),
    'Manganese': joblib.load('model_Mn.joblib'),
    'Boron': joblib.load('model_B.joblib'),
    'Copper': joblib.load('model_Cu.joblib'),
    'Zinc': joblib.load('model_Zn.joblib'),
    'Molybdenum': joblib.load('model_Mo.joblib'),
    'Sodium': joblib.load('model_Na.joblib'),
    'Aluminium': joblib.load('model_Al.joblib')
}

# Load the trained KNN model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.get_json()

    if 'input_x' in input_data and 'input_value' in input_data:
        # Route for nutrient prediction
        input_x = input_data['input_x']
        input_value = float(input_data['input_value'])
        
        # Perform prediction using the selected model
        selected_model = models.get(input_x)
        input_value = np.array([[input_value]])
        prediction = selected_model.predict(input_value)
        
        # Format the output as a dictionary of attribute-value pairs
        attributes = ['humidity', 'water_temp', 'ec', 'ph', 'temp']
        predicted_values = {attr: val for attr, val in zip(attributes, prediction.flatten().tolist())}
        
        return jsonify(predicted_values)
    else:
        # Route for general prediction
        humidity = float(input_data['humidity'])
        water_temp = float(input_data['water_temp'])
        ec = float(input_data['ec'])
        ph = float(input_data['ph'])
        temp = float(input_data['temp'])

        input_values = np.array([[humidity, water_temp, ec, ph, temp]])
        prediction = model.predict(input_values)

        attributes = ['N', 'P', 'K', 'Ca', 'Mg', 'S', 'Fe', 'Mn', 'B', 'Cu', 'Zn', 'Mo', 'Na', 'Al']
        predicted_values = dict(zip(attributes, prediction.flatten().tolist()))

        return jsonify(predicted_values)

if __name__ == '__main__':
    app.run(debug=True)
