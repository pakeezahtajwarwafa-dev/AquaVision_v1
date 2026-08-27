import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold

def engineer_aquatic_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "temperature" in df.columns and "dissolved_oxygen" in df.columns:
        df["temp_do_ratio"] = df["temperature"] / (df["dissolved_oxygen"] + 1e-5)
    if "ammonia" in df.columns and "ph" in df.columns and "temperature" in df.columns:
        df["ammonia_toxicity_index"] = df["ammonia"] * (10 ** (df["ph"] - 7.0)) * (df["temperature"] / 25.0)
    if "ph" in df.columns and "ammonia" in df.columns and "dissolved_oxygen" in df.columns:
        df["stress_score"] = (
            np.abs(df["ph"] - 7.5) * 0.3 +
            (df["ammonia"] * 10.0) * 0.4 +
            np.maximum(0, 6.0 - df["dissolved_oxygen"]) * 0.3
        )
    return df

def generate_kfold_splits(df: pd.DataFrame, n_splits: int = 5) -> pd.DataFrame:
    df = df.copy()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    df["fold"] = -1
    for fold_idx, (_, val_idx) in enumerate(skf.split(df, df["label"])):
        df.loc[val_idx, "fold"] = fold_idx
    return df

def sanitize_and_prepare():
    manifest_dir = Path("datasets/manifests")
    processed_dir = Path("datasets/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    for species in ["fish", "shrimp"]:
        manifest_file = manifest_dir / f"{species}_manifest_deduped.csv"
        if not manifest_file.exists():
            continue

        df = pd.read_csv(manifest_file)
        
        # Strip rows where 'test_split' was ingested as a target class label
        df = df[df["label"] != "test_split"].reset_index(drop=True)

        df_final = generate_kfold_splits(df, n_splits=5)

        wq_file = Path("datasets/water_quality") / f"{species}_water_quality.csv"
        if wq_file.exists():
            df_wq = pd.read_csv(wq_file)
            df_final = pd.merge(df_final, df_wq, on=["image_path", "label"], how="left")
            df_final = engineer_aquatic_features(df_final)

        out_path = processed_dir / f"{species}_processed_manifest.csv"
        df_final.to_csv(out_path, index=False)
        print(f"[{species.upper()}] Sanitized manifest saved: {out_path} ({len(df_final)} samples, {df_final['label'].nunique()} classes)")

if __name__ == "__main__":
    sanitize_and_prepare()
