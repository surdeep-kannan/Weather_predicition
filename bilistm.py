import numpy as np
import pandas as pd
import os
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import Precision, Recall

feature_cols = ['temperature_2m', 'relative_humidity_2m', 'rain', 'surface_pressure']
time_steps = 30
epochs = 15
batch_size = 512
save_folder = "district_models"

if not os.path.exists(save_folder):
    os.makedirs(save_folder)

try:
    print("Loading dataset...")
    df = pd.read_csv('datatrain.csv')
    if 'time' in df.columns: df['time'] = pd.to_datetime(df['time'])
    
    if 'city' in df.columns:
        districts = df['city'].unique()
        print(f"Found {len(districts)} districts in database.")
    else:
        print("Error: 'city' column missing.")
        exit()
except FileNotFoundError:
    print("Error: 'datatrain.csv' not found.")
    exit()

for district in districts:
    model_filename = f"model_{district}_BiLSTM.keras"
    model_path = os.path.join(save_folder, model_filename)
    
    if os.path.exists(model_path):
        print(f"Skipping {district} (Model already exists)")
        continue

    print(f"\nSTARTING TRAINING FOR: {district}")

    # --- ERROR HANDLING START ---
    try:
        city_data = df[df['city'] == district].copy()
        if 'time' in city_data.columns: city_data = city_data.sort_values(by='time')

        city_data['target_next_hour'] = city_data['rain'].shift(-1)
        city_data = city_data.dropna()
        city_data['target_next_hour'] = (city_data['target_next_hour'] > 0.0).astype(int)

        if len(city_data) < (time_steps + 100):
            print(f"Not enough data for {district}. Skipping.")
            continue

        scaler = MinMaxScaler(feature_range=(0, 1))
        city_data[feature_cols] = scaler.fit_transform(city_data[feature_cols])
        
        scaler_path = os.path.join(save_folder, f"scaler_{district}_BiLSTM.pkl")
        joblib.dump(scaler, scaler_path)

        X_vals = city_data[feature_cols].values
        y_vals = city_data['target_next_hour'].values

        X, y = [], []
        for i in range(len(X_vals) - time_steps):
            X.append(X_vals[i:(i + time_steps)])
            y.append(y_vals[i + time_steps])
        X, y = np.array(X), np.array(y)

        weights = class_weight.compute_class_weight('balanced', classes=np.unique(y), y=y)
        weights_dict = dict(enumerate(weights))

        model = Sequential()
        model.add(Input(shape=(X.shape[1], X.shape[2]))) 
        model.add(Bidirectional(LSTM(64, return_sequences=True)))
        model.add(Dropout(0.2))
        model.add(Bidirectional(LSTM(32, return_sequences=False)))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        es = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        model.fit(X, y, epochs=epochs, batch_size=batch_size, 
                  validation_split=0.1, class_weight=weights_dict, callbacks=[es], verbose=1)

        model.save(model_path)
        print(f"Saved: {model_filename}")

    except Exception as e:
        print(f"CRITICAL ERROR TRAINING {district}: {e}")
        print("SKIPPING TO NEXT DISTRICT...")
        continue 

print("\nALL DISTRICTS COMPLETED!")