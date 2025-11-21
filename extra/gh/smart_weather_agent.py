import numpy as np
import pandas as pd
import json
import datetime
import joblib
from tensorflow.keras.models import load_model

# --- CONFIGURATION ---
DISTRICT = "Ariyalur"  # Change this to test different districts
JSON_FILE = "weather_context.json"
MODEL_PATH = f"model_{DISTRICT}.keras"
SCALER_PATH = f"scaler_{DISTRICT}.pkl"

# --- STEP 1: LOAD RESOURCES ---
print(f"🤖 Initializing Agent for {DISTRICT}...")

# Load the Knowledge Base
try:
    with open(JSON_FILE, 'r') as f:
        kb = json.load(f)
    district_info = kb.get(DISTRICT, {})
except FileNotFoundError:
    print("❌ Error: weather_context.json not found.")
    exit()

# Load the AI Brain & Scaler
try:
    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("✅ LSTM Model & Scaler loaded successfully.")
except:
    print("❌ Error: Model or Scaler not found. Did you run training?")
    exit()

# --- STEP 2: GET REAL-TIME CONTEXT ---
current_date = datetime.datetime.now()
current_month = current_date.month
current_season = "Unknown"

# Determine Season dynamically
for season_name, months in district_info.get("seasons", {}).items():
    if current_month in months:
        current_season = season_name
        break

print(f"📅 Date: {current_date.strftime('%Y-%m-%d')} | Season: {current_season}")

# --- STEP 3: SIMULATE INPUT (The Data) ---
# In a real app, you would fetch this from a live sensor or API.
# Here, we create a dummy input representing a "Humid Evening"
print("\n📡 Reading Live Sensor Data...")
input_data = pd.DataFrame({
    'temperature_2m': [28.5],       # 28.5°C
    'relative_humidity_2m': [82.0], # 82% Humidity
    'rain': [0.0],                  # Currently not raining
    'surface_pressure': [1002.0]    # Low pressure (Stormy sign)
})

# Scale the input using the saved scaler (CRITICAL!)
# We must reshape the data to match the LSTM's expected 3D shape (1 sample, 1 step, 4 features)
# Note: Since we trained on 30 steps, a single step prediction is an approximation for this demo
scaled_input = scaler.transform(input_data)
# We duplicate this single reading 30 times to fake a "history" for the LSTM
# (In production, you'd actually use the last 30 hours of real data)
final_input = np.repeat(scaled_input[np.newaxis, :, :], 30, axis=1)

# --- STEP 4: PREDICT & INTERPRET ---
prediction_prob = model.predict(final_input, verbose=0)[0][0]
prediction_percent = int(prediction_prob * 100)
is_raining_soon = prediction_prob > 0.5

print(f"\n🔮 LSTM Raw Probability: {prediction_prob:.4f}")

# --- STEP 5: THE RAG "NOVELTY" LOGIC ---
print("\n📝 GENERATING SMART REPORT:")
print("------------------------------------------------")

report = f"**Weather Forecast for {DISTRICT}**\n"
report += f"Confidence: {prediction_percent}%\n\n"

if is_raining_soon:
    report += "🌧️ PREDICTION: RAIN EXPECTED.\n"
    
    # Context Check: Is rain normal for this season?
    if "Monsoon" in current_season:
        report += f"✅ ANALYSIS: This is typical for the {current_season}. "
        report += f"The low pressure (1002 hPa) confirms the likelihood of a spell."
    elif current_season == "Summer":
        report += f"⚠️ ANOMALY ALERT: Rain is unusual for {current_season}. "
        report += "This could be a localized convectional storm due to high heat."
    else:
        report += f"ℹ️ NOTE: Unseasonal rain detected during {current_season}."

else:
    report += "☀️ PREDICTION: NO RAIN.\n"
    
    # Context Check: Is dry weather normal?
    if "Monsoon" in current_season:
        report += f"⚠️ ANALYSIS: A dry spell is occurring despite it being {current_season}. "
        report += "Check for potential low-pressure system formation in coming days."
    else:
        report += f"✅ ANALYSIS: Consistent with typical {current_season} dry patterns."

# Add District Specific Risk
report += f"\n\n🌍 LOCAL CONTEXT: {district_info.get('risk_profile', '')}"

print(report)
print("------------------------------------------------")