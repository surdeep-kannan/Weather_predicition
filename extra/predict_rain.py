import numpy as np
import tensorflow as tf
import os
import json
from sklearn.preprocessing import StandardScaler
import sys 

# --- Configuration ---
MODEL_DIR = 'trained_models_regression_v2' # Folder where models are saved
SCALER_PARAMS_FILE = 'scaler_params_regression_v2.json' # File with scaling parameters
# FEATURES used in training - MUST be 3 in number and order
FEATURES = ['temperature_2m', 'relative_humidity_2m', 'surface_pressure'] 

def load_and_predict():
    """
    Guides the user through inputting data, loading the appropriate REGRESSION model, 
    and making a rain amount prediction using Temperature, Humidity, and Pressure.
    """
    print("\n--- Today's Rain Prediction Tool (Regression) ---")

    # 1. Load Scaler Parameters
    try:
        with open(SCALER_PARAMS_FILE, 'r') as f:
            scaler_params_all = json.load(f)
        print(f"✅ Scaling parameters loaded from {SCALER_PARAMS_FILE}")
    except FileNotFoundError:
        print(f"\nFATAL ERROR: Scaling parameter file '{SCALER_PARAMS_FILE}' not found.")
        print("Please ensure you ran the training script (train_models_by_district.py) successfully.")
        sys.exit(1)

    # 2. Get District/City Input & Verify Model Exists
    district_name_raw = input("Enter the District/City Name for prediction: ").strip()
    safe_district_name = district_name_raw.replace(" ", "_").replace("/", "_")
    model_filename = os.path.join(MODEL_DIR, f'rain_model_reg_{safe_district_name}.h5')
    
    if safe_district_name not in scaler_params_all:
        print(f"\n⚠️ Error: No scaling data found for '{district_name_raw}'.")
        sys.exit(1)
    
    if not os.path.exists(model_filename):
        print(f"\n⚠️ Error: No trained model file found for '{district_name_raw}' at: {model_filename}")
        sys.exit(1)

    # 3. Configure Scaler and Model
    district_params = scaler_params_all[safe_district_name]
    
    # Recreate the fitted StandardScaler state using saved mean and scale
    scaler = StandardScaler()
    scaler.mean_ = np.array(district_params['mean'])
    scaler.scale_ = np.array(district_params['scale'])
    
    try:
        model = tf.keras.models.load_model(model_filename)
        print(f"✅ Loaded regression model for: {district_name_raw}")
    except Exception as e:
        print(f"FATAL ERROR: Could not load the Keras model. {e}")
        sys.exit(1)

    # 4. Get Feature Input (3 features)
    print("\nPlease enter the current weather conditions:")
    
    try:
        # Collect the three features in the exact order the model expects
        temp = float(input(f"   Enter {FEATURES[0]} (Temperature_2m, e.g., 28.0 C): "))
        humidity = float(input(f"   Enter {FEATURES[1]} (Humidity_2m, e.g., 75.5 %): "))
        pressure = float(input(f"   Enter {FEATURES[2]} (Surface_Pressure, e.g., 1008.5 hPa): ")) 
    except ValueError:
        print("\nInput Error: Please enter valid numerical values for all inputs.")
        sys.exit(1)

    # 5. Prepare and Scale the Data
    
    # Input data MUST be a 2D array: [[temp, humidity, pressure]]
    input_data = np.array([[temp, humidity, pressure]])
    
    # Apply the scaling transformation
    X_scaled = scaler.transform(input_data) 
    
    # 6. Predict
    # Prediction returns a 2D array [[value]], so we index to get the scalar value [0][0]
    predicted_rain_mm = model.predict(X_scaled, verbose=0)[0][0]
    
    # Ensure rain prediction is not negative (clip at zero)
    predicted_rain_mm = max(0, predicted_rain_mm)
    
    # 7. Output Result
    print("\n-------------------------------------------")
    print(f"   District: {district_name_raw}")
    print(f"   Predicted Rain Amount Today (mm): {predicted_rain_mm:.2f} mm")
    
    if predicted_rain_mm >= 10.0:
        print("   -> Prediction: Heavy Rainfall expected today.")
    elif predicted_rain_mm >= 0.1:
        print("   -> Prediction: Light to Moderate Rain expected today.")
    else:
        print("   -> Prediction: Little to No Rain expected today.")
    print("-------------------------------------------")

# Execute the prediction tool
if __name__ == "__main__":
    load_and_predict()