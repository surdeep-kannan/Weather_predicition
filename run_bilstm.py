import pandas as pd
import numpy as np
import requests
import datetime
import os
import joblib
from tensorflow.keras.models import load_model

DISTRICT = "Chennai" 
LIVE_DATA_FILE = "my_friend_data.csv"

BASE_PATH = os.getcwd()
MODEL_PATH = os.path.join(BASE_PATH, "district_models", f"model_{DISTRICT}_BiLSTM.keras")
SCALER_PATH = os.path.join(BASE_PATH, "district_models", f"scaler_{DISTRICT}_BiLSTM.pkl")
FEATURE_COLS = ['temperature_2m', 'relative_humidity_2m', 'rain', 'surface_pressure']

def get_latest_sensor_data():
    try:
        df = pd.read_csv(LIVE_DATA_FILE, header=None)
        latest = df.iloc[-1]
        
        time_str = latest[0]
        dt_obj = datetime.datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
        
        t = float(latest[3])   
        h = float(latest[2])   
        p = float(latest[5])   
        soil = float(latest[6])
        r = 0.0 
        
        return dt_obj, time_str, t, h, p, soil, r
    except Exception as e:
        print(f"Error reading {LIVE_DATA_FILE}: {e}")
        exit()

def get_season_context(dt_obj):
    month = dt_obj.month
    if month in [10, 11, 12]: return "Northeast Monsoon (Rainy Season)"
    elif month in [1, 2]: return "Winter (Dry/Cool)"
    elif month in [3, 4, 5]: return "Summer (Hot)"
    else: return "Southwest Monsoon"

def get_humidity_status(h):
    if h > 70: return "High (Moist Air)"
    elif h > 40: return "Moderate"
    else: return "Low (Dry Air)"

def generate_smart_history(t, h, p, r):
    trend_p = -0.5 if p < 1005 else 0.0
    trend_h = 1.0 if h > 80 else -0.5
    trend_t = 0.5 

    data = {col: [] for col in FEATURE_COLS}
    for i in range(29, -1, -1):
        past_t = t - (trend_t * i) + np.random.normal(0, 0.1)
        past_h = h - (trend_h * i) + np.random.normal(0, 0.5)
        past_p = p - (trend_p * i) + np.random.normal(0, 0.1)
        data['temperature_2m'].append(past_t)
        data['relative_humidity_2m'].append(past_h)
        data['surface_pressure'].append(past_p)
        data['rain'].append(r)
    return pd.DataFrame(data), trend_p

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

if __name__ == "__main__":
    print(f"--- SENSOR ANALYSIS ---")
    
    dt_obj, time_str, temp, hum, press, soil, rain = get_latest_sensor_data()
    season = get_season_context(dt_obj)
    hum_status = get_humidity_status(hum)
    
    print(f"TIME:   {time_str}")
    print(f"SEASON: {season}")
    print(f"SOIL:   {soil}%")
    print(f"DATA:   {temp}C | {hum}% ({hum_status})")

    try:
        model = load_model(MODEL_PATH, compile=False)
        scaler = joblib.load(SCALER_PATH)
    except:
        print("Error: Bi-LSTM files not found.")
        exit()

    history_df, inferred_trend = generate_smart_history(temp, hum, press, rain)
    scaled_input = scaler.transform(history_df[FEATURE_COLS])
    final_input = np.expand_dims(scaled_input, axis=0)

    prob = model.predict(final_input, verbose=0)[0][0]
    confidence = round(prob * 100, 2)
    
    if prob > 0.50:
        if "Winter" in season and hum < 85: 
            forecast = "FOGGY / MISTY"
            action = "MONITOR"
            reason = "High humidity in winter often indicates fog, not heavy rain."
        elif "Summer" in season:
            forecast = "HUMID HEAT"
            action = "CONTINUE IRRIGATION"
            reason = "High humidity is common in summer coastal areas. Rain is unlikely."
        else:
            forecast = "RAIN EXPECTED"
            if soil > 60:
                action = "DELAY IRRIGATION"
                reason = "Soil is wet and rain is coming. Prevent waterlogging."
            else:
                action = "DELAY IRRIGATION"
                reason = f"Soil is dry ({soil}%), but rain is approaching. Wait 2-3 hours to save water."
    else:
        forecast = "CLEAR SKIES"
        if soil < 40:
            action = "START IRRIGATION"
            reason = f"Soil moisture is low ({soil}%) and no rain is predicted."
        else:
            action = "NO ACTION NEEDED"
            reason = "Soil moisture is sufficient."

    print(f"\nVERDICT: {forecast} ({confidence}%) -> {action}")

    prompt = f"""
    You are an Expert Agricultural Scientist.
    
    [CONTEXT]
    - Season: {season}
    - Humidity: {hum}% ({hum_status})
    - Soil Moisture: {soil}%
    
    [DECISION MATRIX]
    - AI Forecast: {forecast}
    - Recommended Action: {action}
    - Scientific Reason: {reason}
    
    [TASK]
    Write a clean, professional Advisory Note for the farmer.
    
    FORMAT:
    **RECOMMENDATION:** [Insert Recommended Action]
    
    **REASONING:** [Explain using the Scientific Reason. Do NOT mention 'probabilities' or 'AI models'. Mention the season and humidity.]
    """

    print("GENERATING ADVISORY...")
    report = ask_llama(prompt)
    
    print("-" * 50)
    print(report.strip())
    print("-" * 50)