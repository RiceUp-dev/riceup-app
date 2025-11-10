from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import os

app = Flask(__name__)
CORS(app)

# Extended dataset from the PDF (2014-2021)
def load_complete_rice_data():
    data = []
    
    # Farmgate prices from Table 1 (2014-2021)
    farmgate_data = [
        # 2014
        {"year": 2014, "quarter": 1, "price": 19.52, "type": "farmgate"},
        {"year": 2014, "quarter": 2, "price": 20.69, "type": "farmgate"},
        {"year": 2014, "quarter": 3, "price": 20.42, "type": "farmgate"},
        {"year": 2014, "quarter": 4, "price": 19.77, "type": "farmgate"},
        # 2015
        {"year": 2015, "quarter": 1, "price": 18.58, "type": "farmgate"},
        {"year": 2015, "quarter": 2, "price": 17.93, "type": "farmgate"},
        {"year": 2015, "quarter": 3, "price": 17.58, "type": "farmgate"},
        {"year": 2015, "quarter": 4, "price": 17.86, "type": "farmgate"},
        # 2016
        {"year": 2016, "quarter": 1, "price": 18.28, "type": "farmgate"},
        {"year": 2016, "quarter": 2, "price": 18.19, "type": "farmgate"},
        {"year": 2016, "quarter": 3, "price": 20.33, "type": "farmgate"},
        {"year": 2016, "quarter": 4, "price": 17.89, "type": "farmgate"},
        # 2017
        {"year": 2017, "quarter": 1, "price": 18.73, "type": "farmgate"},
        {"year": 2017, "quarter": 2, "price": 19.07, "type": "farmgate"},
        {"year": 2017, "quarter": 3, "price": 18.71, "type": "farmgate"},
        {"year": 2017, "quarter": 4, "price": 19.10, "type": "farmgate"},
        # 2018
        {"year": 2018, "quarter": 1, "price": 20.24, "type": "farmgate"},
        {"year": 2018, "quarter": 2, "price": 20.86, "type": "farmgate"},
        {"year": 2018, "quarter": 3, "price": 21.46, "type": "farmgate"},
        {"year": 2018, "quarter": 4, "price": 20.94, "type": "farmgate"},
        # 2019
        {"year": 2019, "quarter": 1, "price": 19.94, "type": "farmgate"},
        {"year": 2019, "quarter": 2, "price": 18.06, "type": "farmgate"},
        {"year": 2019, "quarter": 3, "price": 16.49, "type": "farmgate"},
        {"year": 2019, "quarter": 4, "price": 15.96, "type": "farmgate"},
        # 2020
        {"year": 2020, "quarter": 1, "price": 16.47, "type": "farmgate"},
        {"year": 2020, "quarter": 2, "price": 18.47, "type": "farmgate"},
        {"year": 2020, "quarter": 3, "price": 17.37, "type": "farmgate"},
        {"year": 2020, "quarter": 4, "price": 18.39, "type": "farmgate"},
        # 2021
        {"year": 2021, "quarter": 1, "price": 17.13, "type": "farmgate"},
        {"year": 2021, "quarter": 2, "price": 16.79, "type": "farmgate"},
        {"year": 2021, "quarter": 3, "price": 16.44, "type": "farmgate"},
        {"year": 2021, "quarter": 4, "price": 16.65, "type": "farmgate"},
    ]
    
    # Retail prices from Table 2 (2014-2021)
    retail_data = [
        # 2014
        {"year": 2014, "quarter": 1, "price": 40.60, "type": "retail"},
        {"year": 2014, "quarter": 2, "price": 42.39, "type": "retail"},
        {"year": 2014, "quarter": 3, "price": 43.29, "type": "retail"},
        {"year": 2014, "quarter": 4, "price": 43.04, "type": "retail"},
        # 2015
        {"year": 2015, "quarter": 1, "price": 42.41, "type": "retail"},
        {"year": 2015, "quarter": 2, "price": 41.74, "type": "retail"},
        {"year": 2015, "quarter": 3, "price": 41.10, "type": "retail"},
        {"year": 2015, "quarter": 4, "price": 41.20, "type": "retail"},
        # 2016
        {"year": 2016, "quarter": 1, "price": 41.39, "type": "retail"},
        {"year": 2016, "quarter": 2, "price": 40.05, "type": "retail"},
        {"year": 2016, "quarter": 3, "price": 41.32, "type": "retail"},
        {"year": 2016, "quarter": 4, "price": 41.50, "type": "retail"},
        # 2017
        {"year": 2017, "quarter": 1, "price": 41.79, "type": "retail"},
        {"year": 2017, "quarter": 2, "price": 41.80, "type": "retail"},
        {"year": 2017, "quarter": 3, "price": 41.66, "type": "retail"},
        {"year": 2017, "quarter": 4, "price": 42.33, "type": "retail"},
        # 2018
        {"year": 2018, "quarter": 1, "price": 43.17, "type": "retail"},
        {"year": 2018, "quarter": 2, "price": 44.06, "type": "retail"},
        {"year": 2018, "quarter": 3, "price": 46.36, "type": "retail"},
        {"year": 2018, "quarter": 4, "price": 47.17, "type": "retail"},
        # 2019
        {"year": 2019, "quarter": 1, "price": 46.88, "type": "retail"},
        {"year": 2019, "quarter": 2, "price": 43.21, "type": "retail"},
        {"year": 2019, "quarter": 3, "price": 42.12, "type": "retail"},
        {"year": 2019, "quarter": 4, "price": 41.90, "type": "retail"},
        # 2020
        {"year": 2020, "quarter": 1, "price": 41.45, "type": "retail"},
        {"year": 2020, "quarter": 2, "price": 42.29, "type": "retail"},
        {"year": 2020, "quarter": 3, "price": 42.04, "type": "retail"},
        {"year": 2020, "quarter": 4, "price": 41.57, "type": "retail"},
        # 2021
        {"year": 2021, "quarter": 1, "price": 42.33, "type": "retail"},
        {"year": 2021, "quarter": 2, "price": 42.17, "type": "retail"},
        {"year": 2021, "quarter": 3, "price": 42.39, "type": "retail"},
        {"year": 2021, "quarter": 4, "price": 42.92, "type": "retail"},
    ]
    
    # Combine all data
    all_data = farmgate_data + retail_data
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    
    # Create time index (quarters since Q1 2014)
    df['time_index'] = (df['year'] - 2014) * 4 + df['quarter'] - 1
    
    return df

# Initialize models
models = {}
scalers = {}

def train_models():
    df = load_complete_rice_data()
    print("Training models with data from 2014-2021...")
    print(f"Total data points: {len(df)}")
    
    for price_type in ['farmgate', 'retail']:
        type_data = df[df['type'] == price_type].copy()
        
        if len(type_data) > 0:
            X = type_data[['time_index']].values
            y = type_data['price'].values
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train model
            model = LinearRegression()
            model.fit(X_scaled, y)
            
            models[price_type] = model
            scalers[price_type] = scaler
            
            print(f"{price_type.upper()} Model trained - R²: {model.score(X_scaled, y):.3f}")

# Train models on startup
train_models()

@app.route('/')
def home():
    return jsonify({
        "message": "Rice Price Prediction API", 
        "status": "active",
        "data_years": "2014-2021",
        "available_types": ["farmgate", "retail"]
    })

@app.route('/predict', methods=['POST'])
def predict_price():
    try:
        data = request.get_json()
        price_type = data.get('type', 'retail')
        quarters_ahead = data.get('quarters_ahead', 1)
        
        if price_type not in models:
            return jsonify({"error": "Invalid price type. Use 'farmgate' or 'retail'"}), 400
        
        # Calculate future time index (Q4 2021 is index 31)
        current_index = 31
        future_index = current_index + quarters_ahead
        
        # Scale and predict
        future_index_scaled = scalers[price_type].transform([[future_index]])
        prediction = models[price_type].predict(future_index_scaled)[0]
        
        # Calculate prediction period
        future_year = 2022 + (quarters_ahead - 1) // 4
        future_quarter = ((quarters_ahead - 1) % 4) + 1
        
        return jsonify({
            "predicted_price": round(prediction, 2),
            "price_type": price_type,
            "quarters_ahead": quarters_ahead,
            "prediction_period": f"{future_year}-Q{future_quarter}"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/prices/current', methods=['GET'])
def get_current_prices():
    current_prices = {
        "farmgate": {"price": 16.65, "period": "2021-Q4", "unit": "PHP/kg"},
        "retail": {"price": 42.92, "period": "2021-Q4", "unit": "PHP/kg"}
    }
    return jsonify(current_prices)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "models_loaded": list(models.keys())})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)