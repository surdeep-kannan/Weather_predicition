import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from time import time
import os
import json 

# --- Configuration ---
FILE_PATH = 'datatrain.csv'  
DISTRICT_COLUMN = 'city'     
# UPDATED FEATURES: Now includes surface_pressure
FEATURES = ['temperature_2m', 'relative_humidity_2m', 'surface_pressure'] 
TARGET = 'rain' # Target remains 'rain' (amount of rain today)
MODEL_DIR = 'trained_models_regression_v2' # New folder to store models with pressure data
SCALER_PARAMS_FILE = 'scaler_params_regression_v2.json' # New file for scaling parameters

# Define the columns we need to load (Features + Target + District)
COLUMNS_TO_LOAD = FEATURES + [TARGET] + [DISTRICT_COLUMN] 

# ----------------------------------------------------
## 1. GPU Check and Model Definition
# ----------------------------------------------------
print("--- 1. Checking GPU Status & Setup ---")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print(f"✅ TensorFlow found GPU: {gpus[0].name}")
    except RuntimeError as e:
        print(f"Runtime error with GPU setup: {e}")
else:
    print("⚠️ No GPU found. Training will proceed on CPU.")

# Ensure model saving directory exists
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

def create_model():
    """Defines the Keras model architecture for REGRESSION (3 input features)."""
    model = tf.keras.Sequential([
        # Input layer must match the number of features (3: temp, humidity, pressure)
        tf.keras.layers.Dense(64, activation='relu', input_shape=(len(FEATURES),)),
        tf.keras.layers.Dropout(0.2), 
        tf.keras.layers.Dense(32, activation='relu'),
        # Linear activation for REGRESSION (predicting a continuous value: rain amount)
        tf.keras.layers.Dense(1, activation='linear') 
    ])
    # Use Mean Squared Error (MSE) for regression loss
    model.compile(optimizer='adam', loss='mse', metrics=['mae']) 
    return model

# ----------------------------------------------------
## 2. Efficient Data Loading and Grouping
# ----------------------------------------------------
print("\n--- 2. Loading and Grouping Data ---")
try:
    # Load data by columns for efficiency
    df = pd.read_csv(FILE_PATH, usecols=COLUMNS_TO_LOAD)
except FileNotFoundError:
    print(f"FATAL ERROR: File not found at {FILE_PATH}. Check the file name and directory.")
    exit()

df.dropna(inplace=True)
print(f"Total dataset shape after cleaning: {df.shape}")

district_groups = df.groupby(DISTRICT_COLUMN)
all_districts = list(district_groups.groups.keys())
print(f"Found {len(all_districts)} unique districts. Preparing to train one REGRESSION model for each.")

# Dictionary to store scaler parameters for all districts
scaler_params = {}

# ----------------------------------------------------
## 3. Iterate, Train, and Save Models & Scaler Params
# ----------------------------------------------------
print("\n--- 3. Training and Saving Models & Scaler Parameters ---")
start_total_time = time()

for i, (district_name, group_data) in enumerate(district_groups):
    
    # 3.1 Prepare Data for the current district
    X_district = group_data[FEATURES]
    y_district = group_data[TARGET] 
    
    if X_district.shape[0] < 100:
        print(f"\nProcessing District {i+1}/{len(all_districts)}: {district_name}")
        print(f"   Skipping. Not enough data ({X_district.shape[0]} rows).")
        continue

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_district, y_district, test_size=0.2, random_state=42
    )
    
    # Scale features: IMPORTANT to fit the scaler only on X_train
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save the fitted scaler parameters (mean and standard deviation)
    safe_district_name = district_name.replace(" ", "_").replace("/", "_")
    scaler_params[safe_district_name] = {
        'mean': scaler.mean_.tolist(),
        'scale': scaler.scale_.tolist()
    }
    
    # 3.2 Define and Train Model
    model = create_model()
    
    train_start = time()
    model.fit(
        X_train_scaled, y_train,
        epochs=5,
        batch_size=32,
        validation_split=0.1,
        verbose=0
    )
    train_time = time() - train_start
    
    # 3.3 Evaluate and Save Model
    # Evaluate with mean absolute error (MAE)
    loss_mse, accuracy_mae = model.evaluate(X_test_scaled, y_test, verbose=0)
    
    model_filename = os.path.join(MODEL_DIR, f'rain_model_reg_{safe_district_name}.h5')
    model.save(model_filename)
    
    print(f"\nProcessing District {i+1}/{len(all_districts)}: {district_name}")
    print(f"   -> Data size: {X_district.shape[0]} rows. Test MAE (mm): {accuracy_mae:.4f}. Time: {train_time:.2f}s")
    print(f"   -> Model saved to: {model_filename}")

# Save all scaler parameters to a JSON file
with open(SCALER_PARAMS_FILE, 'w') as f:
    json.dump(scaler_params, f, indent=4)
print(f"\n✅ All scaler parameters saved to: {SCALER_PARAMS_FILE}")

print(f"\n--- Training Complete ---")
print(f"Total time for all districts: {time() - start_total_time:.2f} seconds")
print(f"All models have been processed and saved in the '{MODEL_DIR}' folder.")
