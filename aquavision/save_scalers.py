import joblib
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

checkpoints_dir = Path("checkpoints")
checkpoints_dir.mkdir(parents=True, exist_ok=True)

for species in ["fish", "shrimp"]:
    manifest_path = Path("datasets/processed") / f"{species}_processed_manifest.csv"
    if not manifest_path.exists():
        continue
    
    df = pd.read_csv(manifest_path)
    tab_cols = [c for c in ["temperature", "ph", "dissolved_oxygen", "ammonia", "temp_do_ratio", "ammonia_toxicity_index", "stress_score"] if c in df.columns]
    
    if tab_cols:
        scaler = StandardScaler()
        scaler.fit(df[tab_cols].fillna(0.0))
        scaler_path = checkpoints_dir / f"{species}_scaler.pkl"
        joblib.dump(scaler, scaler_path)
        print(f"[{species.upper()}] Scaler successfully saved to {scaler_path}")
