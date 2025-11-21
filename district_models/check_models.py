import os
import joblib
import json
import numpy as np

# Path to your models
FOLDER_PATH = "district_models"

print(f"📂 Scanning {FOLDER_PATH} for Bi-LSTM Scalers...\n")

if not os.path.exists(FOLDER_PATH):
    print("❌ Error: Folder not found.")
    exit()

scaler_data = {}

# Loop through every file in the folder
for filename in os.listdir(FOLDER_PATH):
    # We only want the BiLSTM scalers (to avoid mixing with old ones)
    if filename.startswith("scaler_") and "_BiLSTM.pkl" in filename:
        
        # Extract district name (e.g., "scaler_Chennai_BiLSTM.pkl" -> "Chennai")
        district_name = filename.replace("scaler_", "").replace("_BiLSTM.pkl", "")
        
        file_path = os.path.join(FOLDER_PATH, filename)
        
        try:
            # Load the scaler
            scaler = joblib.load(file_path)
            
            # Extract MinMaxScaler values (The correct ones!)
            # We cast to list so it can be printed as JSON
            scaler_data[district_name] = {
                "min": scaler.data_min_.tolist(),
                "max": scaler.data_max_.tolist()
            }
            print(f"✅ Processed: {district_name}")
            
        except Exception as e:
            print(f"⚠️ Could not load {filename}: {e}")

# --- PRINT THE FINAL JSON ---
print("\n" + "="*60)
print("👇 COPY AND PASTE THIS INTO YOUR PYTHON SCRIPT / APP 👇")
print("="*60)
print("SCALER_PARAMS = " + json.dumps(scaler_data, indent=4))
print("="*60)

print(f"\nSuccessfully generated parameters for {len(scaler_data)} districts.")
if len(scaler_data) < 38:
    print("⚠️ NOTE: You have fewer districts than expected.")
    print("   You may need to re-run 'train_all_bilstm.py' to finish the rest.")