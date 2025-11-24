import os
import datetime
from contextlib import asynccontextmanager
from typing import Tuple, List, Dict, Any
# === FIX: ADD MISSING JSON IMPORT ===
import json 

import numpy as np
import pandas as pd
import joblib
import httpx

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# TensorFlow is a large dependency. On Vercel, it's safer to attempt import
# conditionally or rely on a successful Vercel build environment.
try:
    from tensorflow.keras.models import load_model
    TENSORFLOW_AVAILABLE = True
except ImportError:
    print("WARNING: TensorFlow not found. Prediction models will be disabled.")
    TENSORFLOW_AVAILABLE = False
except Exception as e:
    print(f"WARNING: TensorFlow load failed: {e}. Prediction models disabled.")
    TENSORFLOW_AVAILABLE = False


# --- FIREBASE ---
import firebase_admin
from firebase_admin import credentials, db

# ==========================================
# CONFIG
# ==========================================
DEFAULT_DISTRICT = "Chennai"
# Use relative path for Vercel deployment
BASE_PATH = os.getcwd() 
MODELS_DIR = os.path.join(BASE_PATH, "district_models")
FEATURE_COLS = ['temperature_2m', 'relative_humidity_2m', 'rain', 'surface_pressure']

# NOTE: For Vercel deployment, FIREBASE_KEY_PATH should be handled via a secure
# method like Environment Variables storing the JSON content, or disabled entirely.
# We disable it here for safe Vercel deployment since the key is sensitive.
FIREBASE_KEY_PATH = "agriweather_key.json" # Still referenced, but init is protected below
FIREBASE_DATABASE_URL = "https://agriweather-e5ba4-default-rtdb.firebaseio.com"
FIREBASE_DATA_PATH = "sensor_readings"

# NOTE: Localhost cannot be used on Vercel. This is disabled for safety.
LLAMA_API_URL = os.environ.get("LLAMA_API_URL", "http://llama-is-offline.local") 
LLAMA_MODEL_NAME = "llama3.2:1b"

ai_resources = {}
loaded_models = {}

class ChatRequest(BaseModel):
    message: str
    district: str = DEFAULT_DISTRICT
    # Assuming chat history is NOT needed for this presentation-ready serverless function

# ==========================================
# MODEL LOADER
# ==========================================
def get_model_resources(district_name: str):
    if not TENSORFLOW_AVAILABLE:
        return None
        
    if district_name in loaded_models:
        return loaded_models[district_name]

    model_path = os.path.join(MODELS_DIR, f"model_{district_name}_BiLSTM.keras")
    scaler_path = os.path.join(MODELS_DIR, f"scaler_{district_name}_BiLSTM.pkl")

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        # Fallback to default district model if available
        if district_name != DEFAULT_DISTRICT:
            return get_model_resources(DEFAULT_DISTRICT)
        return None

    resources = {
        "model": load_model(model_path, compile=False),
        "scaler": joblib.load(scaler_path)
    }

    loaded_models[district_name] = resources
    return resources

# ==========================================
# FIREBASE READERS (NO MOCKS)
# ==========================================
def _parse_iso_ts(ts_str: str) -> datetime.datetime:
    """Parse ISO timestamp from stored keys or Time_ISO value."""
    try:
        if isinstance(ts_str, str) and ts_str.endswith("Z"):
            return datetime.datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        return datetime.datetime.fromisoformat(ts_str)
    except Exception:
        return datetime.datetime.utcnow()

def get_all_readings_and_prepare_model_input() -> Tuple[pd.DataFrame, dict, str, List[Tuple[str, Dict[str, Any]]]]:
    """
    Reads ALL timestamped readings from Firebase, sorts them chronologically.
    
    Returns:
        df: pd.DataFrame with columns FEATURE_COLS and length 30 (the latest sequence)
        latest: dict for the latest reading
        latest_ts: timestamp string key for the latest reading
        all_items: List of (timestamp, reading_dict) for all history
        
    Raises HTTPException on problems.
    """
    if not ai_resources.get("firebase_initialized"):
        raise HTTPException(status_code=503, detail="Firebase not connected or models missing. Cannot retrieve sensor data.")

    try:
        ref = db.reference(FIREBASE_DATA_PATH)
        # Fetch ALL data available in the path
        snapshot = ref.get()

        if not snapshot or not isinstance(snapshot, dict):
            raise HTTPException(status_code=503, detail="Firebase path empty or malformed. Expected timestamp-keyed child nodes.")

        # snapshot keys should be timestamps; sort them lexicographically (ISO 8601 sorts chronologically)
        items = sorted(snapshot.items(), key=lambda kv: kv[0])
        
        # Check if enough data is available for the model's sequence length (30)
        if len(items) < 30:
            raise HTTPException(status_code=503, detail=f"The BiLSTM model requires an input sequence of 30 historical readings. Found only {len(items)}. Populate Firebase with more historical readings.")

        # Slice the last 30 items for the model input sequence (fixed-length sequence required by the BiLSTM model).
        last_30 = items[-30:]
        
        temps, hums, pres, rains = [], [], [], []
        soil_vals = []

        for ts_key, entry in last_30:
            if not isinstance(entry, dict):
                raise HTTPException(status_code=500, detail=f"Malformed entry at {ts_key}. Expected dict.")
            # Convert values — if missing or non-convertible raise an informative error
            try:
                temps.append(float(entry.get("Temp_DHT")))
                hums.append(float(entry.get("Hum_DHT")))
                pres.append(float(entry.get("Pressure_hPa")))
                rains.append(float(entry.get("Rain", 0.0)))
                soil_vals.append(float(entry.get("Soil_Moisture_Raw", 0.0)))
            except Exception as ex:
                raise HTTPException(status_code=500, detail=f"Invalid numeric sensor value at {ts_key}: {ex}")

        df = pd.DataFrame({
            "temperature_2m": temps,
            "relative_humidity_2m": hums,
            "surface_pressure": pres,
            "rain": rains
        })

        latest_ts, latest_entry = last_30[-1][0], last_30[-1][1]
        # Attach Time_ISO if not already present
        latest_entry = dict(latest_entry)
        latest_entry["Time_ISO"] = latest_entry.get("Time_ISO", latest_ts)

        return df, latest_entry, latest_ts, items

    except HTTPException:
        # pass through our own raised HTTPException
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Firebase read error: {e}")

# ==========================================
# SERVER LIFESPAN (Vercel uses this on cold start)
# ==========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize Firebase once
    try:
        # Check if the private key content is available via Environment Variable
        if "FIREBASE_PRIVATE_KEY_JSON" in os.environ and not firebase_admin._apps:
            cred_json_str = os.environ["FIREBASE_PRIVATE_KEY_JSON"]
            cred_json = json.loads(cred_json_str)
            cred = credentials.Certificate(cred_json)
            firebase_admin.initialize_app(cred, {"databaseURL": FIREBASE_DATABASE_URL})
            ai_resources["firebase_initialized"] = True
            print("Firebase initialized using environment variable.")
        else:
            ai_resources["firebase_initialized"] = False
            print("Firebase disabled: Missing FIREBASE_PRIVATE_KEY_JSON environment variable or already initialized.")
            
    except Exception as e:
        ai_resources["firebase_initialized"] = False
        print("Firebase init failed:", e)

    # Preload default model (optional)
    _ = get_model_resources(DEFAULT_DISTRICT)
    yield
    loaded_models.clear()

# Vercel looks for the 'app' variable in this module.
app = FastAPI(title="Agri-Weather AI", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# HELPERS
# ==========================================
def get_season_context(dt: datetime.datetime) -> str:
    m = dt.month
    if m in [10, 11, 12]:
        return "Northeast Monsoon"
    if m in [1, 2]:
        return "Winter"
    if m in [3, 4, 5]:
        return "Summer"
    return "Southwest Monsoon"

def get_daily_summary(all_items: List[Tuple[str, Dict[str, Any]]], latest_ts: str) -> str:
    """Calculates summary statistics for the current day from all available readings."""
    if not all_items:
        return "No historical data to summarize."

    latest_date = _parse_iso_ts(latest_ts).date()
    daily_readings = []

    for ts_key, entry in all_items:
        try:
            ts_date = _parse_iso_ts(ts_key).date()
            # Filter all data points that occurred on the current day
            if ts_date == latest_date:
                daily_readings.append({
                    "temp": float(entry.get("Temp_DHT", np.nan)),
                    "hum": float(entry.get("Hum_DHT", np.nan)),
                    "soil": float(entry.get("Soil_Moisture_Raw", np.nan)),
                    "rain": float(entry.get("Rain", np.nan))
                })
        except Exception:
            continue # Skip malformed entries

    if not daily_readings:
        return f"No data found for the current day ({latest_date})."
    
    # Filter out entries where critical values are NaN
    df_daily = pd.DataFrame(daily_readings).dropna(subset=['temp', 'hum', 'soil', 'rain'])
    
    if df_daily.empty:
        return f"No valid numeric data found for the current day ({latest_date})."

    t_avg, t_min, t_max = df_daily['temp'].agg(['mean', 'min', 'max']).round(1)
    h_avg, h_min, h_max = df_daily['hum'].agg(['mean', 'min', 'max']).round(1)
    # Use the overall minimum soil moisture for the day as the critical metric
    soil_min = df_daily['soil'].min().round(1) 
    total_rain = df_daily['rain'].sum().round(2)
    count = len(df_daily)

    return (
        f"Daily Summary (Total {count} readings today, starting from {latest_date}): "
        f"Temp Avg/Min/Max: {t_avg}°C/{t_min}°C/{t_max}°C. "
        f"Humidity Avg/Min/Max: {h_avg}%/{h_min}%/{h_max}%. "
        f"Minimum Soil Moisture recorded: {soil_min}%. "
        f"Total Daily Rainfall (Accumulated): {total_rain}mm."
    )

async def ask_ai(prompt: str) -> str:
    # Check if the LLAMA_API_URL is still local or default (i.e., not configured for deployment)
    if 'localhost' in LLAMA_API_URL or 'llama-is-offline.local' in LLAMA_API_URL:
        return "**RECOMMENDATION:** AI chat is offline.\n**REASONING:** Llama server URL is not configured for remote access."

    payload = {
        "model": LLAMA_MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            # --- REFINED SYSTEM INSTRUCTION ---
            "system": (
                "You are a highly concise Agronomist AI, specializing in real-time sensor data interpretation. "
                "You MUST use the provided context, including the FULL DAILY HISTORY ANALYSIS, to inform your answer. "
                "NO greetings, NO fillers, ONLY direct, professional agricultural advisory."
            )
            # ---------------------------------
        }
    }

    try:
        async with httpx.AsyncClient() as client:
            # Use the environment variable-set URL
            r = await client.post(LLAMA_API_URL, json=payload, timeout=45)
            r.raise_for_status()
            return r.json().get("response", "").strip()
    except Exception:
        return "**RECOMMENDATION:** AI engine offline.\n**REASONING:** Could not reach the configured LLama server."

# ==========================================
# API — ADVISORY
# ==========================================
@app.get("/api/agri-advisory")
async def get_advisory(district: str = Query(DEFAULT_DISTRICT)):
    """
    Returns advisory using the last 30 real readings for the model, and all historical data
    for the LLM context.
    """
    # Load all historical readings and prepare the last 30 for the model input
    # This will raise a 503 if Firebase is not connected or data is insufficient
    hist_df, latest_entry, latest_ts, all_items = get_all_readings_and_prepare_model_input()
    season = get_season_context(_parse_iso_ts(latest_entry.get("Time_ISO", latest_ts)))

    model_res = get_model_resources(district)
    
    # If model resources are unavailable, we cannot provide prediction, but can still advise.
    if model_res is None:
        # Fallback to a rule-based system if the ML model/TF is missing
        prob = 0.0
        conf = 50.0 # Low confidence for rule-based
    else:
        # Scale and reshape for BiLSTM: scaler expects same column order as FEATURE_COLS
        try:
            scaled = model_res["scaler"].transform(hist_df[FEATURE_COLS])
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Scaler transform failed: {e}")

        final = np.expand_dims(scaled, axis=0)  # shape -> (1, 30, 4)

        try:
            prob = float(model_res["model"].predict(final, verbose=0)[0][0])
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Model prediction failed: {e}")

        conf = round(prob * 100, 2)

    # Extract latest sensor values
    try:
        t = float(latest_entry.get("Temp_DHT"))
        h = float(latest_entry.get("Hum_DHT"))
        p = float(latest_entry.get("Pressure_hPa"))
        soil = float(latest_entry.get("Soil_Moisture_Raw", 0.0))
        r = float(latest_entry.get("Rain", 0.0))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Invalid latest sensor numeric values: {e}")

    # Decision logic (kept from your original)
    if soil <= 5:
        forecast = "CRITICAL DROUGHT"
        action = "EMERGENCY IRRIGATION"
        reason = f"Soil moisture extremely low at {soil}%."
    elif prob > 0.30:
        if "Winter" in season and h < 85:
            forecast = "FOGGY"
            action = "MONITOR"
            reason = "High humidity + winter fog signatures."
        elif "Summer" in season:
            forecast = "HUMID HEAT"
            action = "CONTINUE IRRIGATION"
            reason = "Summer humidity ≠ rain."
        else:
            forecast = "RAIN LIKELY"
            action = "DELAY IRRIGATION" if soil < 50 else "STOP IRRIGATION"
            reason = f"Rain chance {conf}%."
    else:
        forecast = "LOW CHANCE OF RAIN"
        action = "START IRRIGATION" if soil < 40 else "NO ACTION"
        reason = f"Soil {soil}% and low rainfall probability."

    # Use full data for LLM advisory
    daily_summary = get_daily_summary(all_items, latest_ts)
    
    prompt = f"""
    --- FULL DATA ANALYSIS CONTEXT ---
    Model Forecast (Rain Probability): {conf}%
    {daily_summary}
    ----------------------------------
    **RECOMMENDATION:** {action}
    **REASONING:** {reason}
    """

    ai_advice = await ask_ai(prompt)

    return {
        "timestamp": latest_entry.get("Time_ISO", latest_ts),
        "location": district,
        "season": season,
        "sensors": {
            "temperature": t,
            "humidity": h,
            "pressure": p,
            "soil_moisture": soil,
            "rain": r
        },
        "analysis": {
            "forecast": forecast,
            "confidence": conf,
            "action": action,
            "reason": reason
        },
        "llm_advisory": ai_advice
    }

# ==========================================
# API — CHAT
# ==========================================
@app.post("/api/chat")
async def chat(req: ChatRequest):
    
    # Fetch all historical readings and get the latest entry
    # This will raise a 503 if Firebase is not connected or data is insufficient
    _, latest_entry, latest_ts, all_items = get_all_readings_and_prepare_model_input()
    season = get_season_context(_parse_iso_ts(latest_entry.get("Time_ISO", latest_ts)))

    daily_summary = get_daily_summary(all_items, latest_ts)

    try:
        t = float(latest_entry.get("Temp_DHT"))
        h = float(latest_entry.get("Hum_DHT"))
        p = float(latest_entry.get("Pressure_hPa"))
        soil = float(latest_entry.get("Soil_Moisture_Raw", 0.0))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Invalid latest sensor numeric values: {e}")

    # --- REFINED CHAT PROMPT: Includes FULL DAILY HISTORY ANALYSIS ---
    prompt = f"""
    --- LIVE SENSOR DATA FOR {req.district.upper()} ---
    Current Season: {season}
    Temperature (Temp_DHT): {t}°C
    Relative Humidity (Hum_DHT): {h}%
    Surface Pressure (Pressure_hPa): {p} hPa
    Soil Moisture (Soil_Moisture_Raw): {soil}%
    
    --- DAILY HISTORY ANALYSIS ---
    {daily_summary}
    --------------------------------------------------

    You MUST analyze the LIVE SENSOR DATA and the DAILY HISTORY ANALYSIS above and provide a concise, immediate agricultural status report.
    
    1. Focus ONLY on the **current, immediate agricultural implication** of the sensor readings within the context of the {season} season and the daily history.
    2. Do NOT provide speculative forecasts (like "rain may occur").
    3. If the user's question is vague (like a greeting or a statement of facts), treat it as a request for the current Agricultural Status Report.
    4. Base your entire response on the current data and the season.
    
    User question: {req.message}
    """
    # --------------------------------------------------------------------------------

    reply = await ask_ai(prompt)

    return {"reply": reply}