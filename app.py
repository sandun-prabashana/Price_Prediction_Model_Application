from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)
CORS(app)

# Load model and scaler
with open('models/model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

with open('models/scaler.pkl', 'rb') as scaler_file:
    loaded_scaler = pickle.load(scaler_file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Extract features from form and create a DataFrame
            input_data = {
                'Baths': [float(request.form['Baths'])],
                'Beds': [float(request.form['Beds'])],
                'District_  Colombo': [float(request.form['District_Colombo'])],
                'District_  Gampaha': [float(request.form['District_Gampaha'])],
                'House size': [float(request.form['House_size'])],
                'Land size': [float(request.form['Land_size'])],
                'Lat': [float(request.form['Lat'])],
                'Lon': [float(request.form['Lon'])],
                'has_cctv': [float(request.form['has_cctv'])],
                'has_garage': [float(request.form['has_garage'])],
                'has_garden': [float(request.form['has_garden'])],
                'has_pool': [float(request.form['has_pool'])],
                'story': [float(request.form['story'])]
            }
            sample_data_original_scale = pd.DataFrame(input_data)
            
            print(input_data)

            # Log-transform and scale the variables
            skewed_cols_adjusted = ['Baths', 'Beds', 'House size', 'Land size', 'Lat', 'Lon', 'story']
            sample_data_log_transformed = sample_data_original_scale.copy()
            sample_data_log_transformed[skewed_cols_adjusted] = np.log1p(sample_data_log_transformed[skewed_cols_adjusted])
            sample_data_log_transformed_scaled = loaded_scaler.transform(sample_data_log_transformed)

            # Make a prediction
            predicted_price_log_transformed = loaded_model.predict(sample_data_log_transformed_scaled)
            actual_predicted_price_log_transformed = np.expm1(predicted_price_log_transformed[0])
            
            # Render the result
            return jsonify({'status': 'success', 'prediction': actual_predicted_price_log_transformed}), 200

        except ValueError:
            return jsonify({'status': 'error', 'message': 'Please enter valid numbers.'}), 400
        
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
