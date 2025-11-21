import numpy as np
import pandas as pd
import os
import joblib  # To save the Scalers
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import class_weight  # KEY FIX: For Class Weights
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import Precision, Recall # KEY FIX: Better metrics

# ==========================================
# 1. CONFIGURATION
# ==========================================
# The 4 physics features you selected
feature_cols = ['temperature_2m', 'relative_humidity_2m', 'rain', 'surface_pressure']
target_col = 'rain_tomorrow' 

# Look back 30 steps (e.g., 30 hours) to predict the next one
time_steps = 30
epochs = 15  
batch_size = 512

# Create folder to save models
save_folder = "district_models"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# ==========================================
# 2. LOAD DATA
# ==========================================
try:
    print("Loading dataset...")
    # REPLACE WITH YOUR EXACT FILE NAME
    df = pd.read_csv('datatrain.csv') 
    
    # Ensure time is sorted correctly
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
    
    print(f"Total Data Loaded: {len(df)} rows")
except FileNotFoundError:
    print("❌ ERROR: 'weather_data.csv' not found. Please check the file name.")
    exit()

# ==========================================
# 3. TRAINING LOOP (PER DISTRICT)
# ==========================================
# Get list of all unique cities
districts = df['city'].unique()
print(f"Found {len(districts)} districts. Starting training...")

for city in districts:
    print(f"\n------------------------------------------------")
    print(f"📍 PROCESSING: {city}")
    
    # --- A. ISOLATE CITY DATA ---
    city_data = df[df['city'] == city].copy()
    city_data = city_data.sort_values(by='time')
    
    # Check for sufficient data (need at least time_steps + some for training)
    if len(city_data) < (time_steps + 100):
        print(f"⚠️ Skipping {city}: Not enough data ({len(city_data)} rows)")
        continue

    # --- B. SCALE DATA (Create the math logic for THIS city) ---
    scaler = MinMaxScaler(feature_range=(0, 1))
    city_data[feature_cols] = scaler.fit_transform(city_data[feature_cols])
    
    # Save Scaler immediately
    # (You NEED this later to make predictions for this specific city)
    scaler_filename = os.path.join(save_folder, f"scaler_{city}.pkl")
    joblib.dump(scaler, scaler_filename)

    # --- C. PREPARE SEQUENCES ---
    X_vals = city_data[feature_cols].values
    y_vals = city_data[target_col].values
    
    X, y = [], []
    for i in range(len(X_vals) - time_steps):
        X.append(X_vals[i:(i + time_steps)])
        y.append(y_vals[i + time_steps])
        
    X = np.array(X)
    y = np.array(y)
    
    # --- D. CALCULATE CLASS WEIGHTS (THE FIX) ---
    # This forces the model to pay attention to RAIN (which is usually rare)
    # It prevents the "70% accuracy" trap where it just guesses "No Rain" alway
    weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y),
        y=y
    )
    weights_dict = dict(enumerate(weights))
    print(f"     Class Weights: {weights_dict}")

    
    model = Sequential()
    model.add(Input(shape=(X.shape[1], X.shape[2]))) 
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(32, return_sequences=False))
    model.add(Dropout(0.2))
    
   
    model.add(Dense(1, activation='sigmoid'))
    
    
    model.compile(
        optimizer='adam', 
        loss='binary_crossentropy', 
        metrics=['accuracy', Precision(name='precision'), Recall(name='recall')]
    )
    
    # --- F. TRAIN ---
    # EarlyStopping saves time if model stops improving
    es = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    
    history = model.fit(
        X, y, 
        epochs=epochs, 
        batch_size=batch_size, 
        validation_split=0.1,
        class_weight=weights_dict, # <--- APPLYING THE WEIGHTS HERE
        callbacks=[es],
        verbose=1
    )
    
    # --- G. SAVE MODEL ---
    # Sanitize name (replace spaces with underscores)
    safe_city_name = str(city).replace(" ", "_")
    model_path = os.path.join(save_folder, f"model_{safe_city_name}.keras")
    model.save(model_path)
    print(f"✅ Model saved: {model_path}")

print("\n================================================")
print("🎉 All districts processed successfully!")
print(f"Models and scalers are in the '{save_folder}' folder.")