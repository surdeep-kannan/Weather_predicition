import pandas as pd
import numpy as np
import requests
import datetime
import os
import joblib
import random
from tensorflow.keras.models import load_model

LAST_RAW_READING = "2025-11-19 10:53:52,28.3,75.3,27.71,64.99,1010.01,31,0"

DISTRICT = "Chennai" 
BASE_PATH = os.getcwd()
MODEL_PATH = os.path.join(BASE_PATH, "district_models", f"model_{DISTRICT}.keras")
SCALER_PATH = os.path.join(BASE_PATH, "district_models", f"scaler_{DISTRICT}.pkl")
FEATURE_COLS = ['temperature_2m', 'relative_humidity_2m', 'rain', 'surface_pressure']

def simulate_next_reading(raw_string):
    parts = raw_string.split(',')
    
    old_time_str = parts[0]
    dt_obj = datetime.datetime.strptime(old_time_str, "%Y-%m-%d %H:%M:%S")
    
    new_time_obj = dt_obj + datetime.timedelta(seconds=2)
    new_time_str = new_time_obj.strftime("%Y-%m-%d %H:%M:%S")
    
    old_temp = float(parts[3])
    old_hum = float(parts[4])
    old_press = float(parts[5])
    
    new_temp = old_temp + random.uniform(-0.05, 0.05)
    new_hum = old_hum + random.uniform(-0.1, 0.1)
    new_press = old_press + random.uniform(-0.02, 0.02)
    
    return {
        "time_obj": new_time_obj,
        "time_str": new_time_str,
        "temp": round(new_temp, 2),
        "hum": round(new_hum, 2),
        "press": round(new_press, 2),
        "rain": 0.0
    }

def generate_smart_history(t, h, p, r):
    trend_p = 0.0 
    trend_h = 0.5 if h > 70 else 0.0
    trend_t = 0.0

    data = {col: [] for col in FEATURE_COLS}
    for i in range(29, -1, -1):
        past_t = t - (trend_t * i) + np.random.normal(0, 0.1)
        past_h = h - (trend_h * i) + np.random.normal(0, 0.5)
        past_p = p - (trend_p * i) + np.random.normal(0, 0.1)
        
        data['temperature_2m'].append(past_t)
        data['relative_humidity_2m'].append(past_h)
        data['surface_pressure'].append(past_p)
        data['rain'].append(r)
        
    return pd.DataFrame(data)

def ask_llama(prompt):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "llama3.2:1b", "prompt": prompt, "stream": False}
        )
        if response.status_code == 200:
            return response.json()['response']
        return "Error: Llama not responding."
    except:
        return "Error: Connection failed."

def get_season(month):
    if month in [10, 11, 12]: return "Northeast Monsoon"
    elif month in [1, 2]: return "Winter"
    elif month in [3, 4, 5]: return "Summer"
    else: return "Southwest Monsoon"

if __name__ == "__main__":
    print(f"⚡ SIMULATING LIVE SENSOR STREAM FOR {DISTRICT}")

    data = simulate_next_reading(LAST_RAW_READING)
    
    print(f"\n🔴 LIVE UPDATE RECEIVED:")
    print(f"   Time: {data['time_str']}") 
    print(f"   Sensors: {data['temp']}°C | {data['hum']}% | {data['press']} hPa")

    try:
        model = load_model(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
    except:
        print("❌ Error: Models not found.")
        exit()

    history_df = generate_smart_history(data['temp'], data['hum'], data['press'], data['rain'])
    scaled_input = scaler.transform(history_df)
    final_input = np.expand_dims(scaled_input, axis=0)

    prob = model.predict(final_input, verbose=0)[0][0]
    verdict = "RAIN LIKELY" if prob > 0.5 else "NO RAIN"
    confidence = round(prob * 100, 2)

    print(f"\n🧠 PHYSICS MODEL VERDICT: {verdict} ({confidence}%)")

    season = get_season(data['time_obj'].month)
    
    prompt = f"""
    You are a Senior Meteorologist for {DISTRICT}, India.
    
    [LIVE SENSOR DATA]
    - Simulation Time: {data['time_str']}
    - Season: {season}
    - Readings: Temp {data['temp']}C | Hum {data['hum']}% | Press {data['press']}hPa
    
    [AI MODEL CALCULATION]
    The Physics Model predicts: {verdict} (Confidence: {confidence}%)
    
    [TASK]
    Write a short forecast for THE REST OF TODAY ({data['time_str']}).
    1. Explicitly state the simulation time (10:53 AM).
    2. Analyze if this aligns with the '{season}' pattern.
    """

    print("📝 GENERATING REPORT...")
    report = ask_llama(prompt)
    
    print("-" * 50)
    print(report.strip())
    print("-" * 50)