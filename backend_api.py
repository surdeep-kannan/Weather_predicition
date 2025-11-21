import os
import datetime
from contextlib import asynccontextmanager
from typing import Tuple

import numpy as np
import pandas as pd
import joblib
import httpx

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from tensorflow.keras.models import load_model

# --- FIREBASE ---
import firebase_admin
from firebase_admin import credentials, db

# ==========================================
# CONFIG
# ==========================================
DEFAULT_DISTRICT = "Chennai"
BASE_PATH = os.getcwd()
MODELS_DIR = os.path.join(BASE_PATH, "district_models")
FEATURE_COLS = ['temperature_2m', 'relative_humidity_2m', 'rain', 'surface_pressure']

FIREBASE_KEY_PATH = "agriweather_key.json"
FIREBASE_DATABASE_URL = "https://agriweather-e5ba4-default-rtdb.firebaseio.com"
FIREBASE_DATA_PATH = "sensor_readings"

LLAMA_API_URL = "http://localhost:11434/api/generate"
LLAMA_MODEL_NAME = "llama3.2:1b"

ai_resources = {}
loaded_models = {}

class ChatRequest(BaseModel):
    message: str
    district: str = DEFAULT_DISTRICT

# ==========================================
# MODEL LOADER
# ==========================================
def get_model_resources(district_name: str):
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

def get_last_30_readings() -> Tuple[pd.DataFrame, dict, str]:
    """
    Read the last 30 timestamped readings from Firebase.
    Expects structure:
    sensor_readings/
        2025-11-20T10:00:00Z: { Temp_DHT: ..., Hum_DHT: ..., Pressure_hPa: ..., Soil_Moisture_Raw: ..., Rain: ... }
        2025-11-20T10:01:00Z: { ... }
        ...
    Returns:
        df: pd.DataFrame with columns FEATURE_COLS and length 30 (oldest → newest)
        latest: dict for the latest reading
        latest_ts: timestamp string key for the latest reading
    Raises HTTPException on problems.
    """
    if not ai_resources.get("firebase_initialized"):
        raise HTTPException(status_code=503, detail="Firebase not connected.")

    try:
        ref = db.reference(FIREBASE_DATA_PATH)
        snapshot = ref.get()

        if not snapshot or not isinstance(snapshot, dict):
            raise HTTPException(status_code=503, detail="Firebase path empty or malformed. Expected timestamp-keyed child nodes.")

        # snapshot keys should be timestamps; sort them lexicographically (ISO 8601 sorts chronologically)
        items = sorted(snapshot.items(), key=lambda kv: kv[0])
        if len(items) < 30:
            raise HTTPException(status_code=503, detail=f"Need at least 30 timestamped readings, found {len(items)}. Populate Firebase with historical readings.")

        last_30 = items[-30:]  # list of (ts_str, entry_dict), oldest → newest among the slice
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

        return df, latest_entry, latest_ts

    except HTTPException:
        # pass through our own raised HTTPException
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Firebase read error: {e}")

# ==========================================
# SERVER LIFESPAN
# ==========================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize Firebase once
    try:
        cred = credentials.Certificate(FIREBASE_KEY_PATH)
        firebase_admin.initialize_app(cred, {"databaseURL": FIREBASE_DATABASE_URL})
        ai_resources["firebase_initialized"] = True
        print("Firebase initialized.")
    except Exception as e:
        ai_resources["firebase_initialized"] = False
        print("Firebase init failed:", e)

    # Preload default model (optional)
    _ = get_model_resources(DEFAULT_DISTRICT)
    yield
    loaded_models.clear()

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

async def ask_ai(prompt: str) -> str:
    payload = {
        "model": LLAMA_MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            # --- REFINED SYSTEM INSTRUCTION ---
            "system": (
                "You are a highly concise Agronomist AI, specializing in real-time sensor data interpretation. "
                "You MUST use the provided context to inform your answer. "
                "NO greetings, NO fillers, ONLY direct, professional agricultural advisory."
            )
            # ---------------------------------
        }
    }

    try:
        async with httpx.AsyncClient() as client:
            r = await client.post(LLAMA_API_URL, json=payload, timeout=45)
            r.raise_for_status()
            return r.json().get("response", "").strip()
    except Exception:
        return "**RECOMMENDATION:** AI engine offline.\n**REASONING:** Llama server unreachable."

# ==========================================
# API — ADVISORY
# ==========================================
@app.get("/api/agri-advisory")
async def get_advisory(district: str = Query(DEFAULT_DISTRICT)):
    """
    Returns advisory using the last 30 real readings from Firebase (no mocks).
    Requires at least 30 timestamped entries in sensor_readings/.
    """
    # Load last 30 readings and latest entry
    hist_df, latest_entry, latest_ts = get_last_30_readings()
    season = get_season_context(_parse_iso_ts(latest_entry.get("Time_ISO", latest_ts)))

    model_res = get_model_resources(district)
    if model_res is None:
        raise HTTPException(status_code=500, detail="Model missing for requested district and default fallback.")

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

    prompt = f"""
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
    
    _, latest_entry, latest_ts = get_last_30_readings()
    season = get_season_context(_parse_iso_ts(latest_entry.get("Time_ISO", latest_ts)))

    try:
        t = float(latest_entry.get("Temp_DHT"))
        h = float(latest_entry.get("Hum_DHT"))
        p = float(latest_entry.get("Pressure_hPa"))
        soil = float(latest_entry.get("Soil_Moisture_Raw", 0.0))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Invalid latest sensor numeric values: {e}")

    # --- REFINED CHAT PROMPT: Highly restrictive to force immediate data analysis ---
    prompt = f"""
    --- LIVE SENSOR DATA FOR {req.district.upper()} ---
    Current Season: {season}
    Temperature (Temp_DHT): {t}°C
    Relative Humidity (Hum_DHT): {h}%
    Surface Pressure (Pressure_hPa): {p} hPa
    Soil Moisture (Soil_Moisture_Raw): {soil}%
    --------------------------------------------------

    You MUST analyze the LIVE SENSOR DATA above and provide a concise, immediate agricultural status report.
    
    1. Focus ONLY on the **current, immediate agricultural implication** of the sensor readings within the context of the {season} season.
    2. Do NOT provide speculative forecasts (like "rain may occur").
    3. If the user's question is vague (like a greeting or a statement of facts), treat it as a request for the current Agricultural Status Report.
    4. Base your entire response on the current data and the season.
    
    User question: {req.message}
    """
    # --------------------------------------------------------------------------------

    reply = await ask_ai(prompt)

    return {"reply": reply}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)