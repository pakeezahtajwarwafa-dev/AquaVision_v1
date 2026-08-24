from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


def generate_synthetic_water_data(df_manifest: pd.DataFrame, noise_rate: float = 0.07, seed: int = 42) -> pd.DataFrame:
    """
    Generates synthetic water quality parameters (pH, Temp, DO, Salinity, Ammonia, Turbidity)
    correlated with image sample disease labels.
    """
    np.random.seed(seed)
    records = []

    for _, row in df_manifest.iterrows():
        is_healthy = "healthy" in row["label"].lower()
        
        if is_healthy:
            ph = np.random.normal(7.2, 0.3)
            temp = np.random.normal(26.0, 1.5)
            do = np.random.normal(6.5, 0.5)
            salinity = np.random.normal(15.0, 2.0)
            ammonia = np.random.exponential(0.02)
            turbidity = np.random.normal(12.0, 3.0)
        else:
            ph = np.random.normal(6.1, 0.6)
            temp = np.random.normal(31.0, 2.5)
            do = np.random.normal(3.2, 0.8)
            salinity = np.random.normal(28.0, 4.0)
            ammonia = np.random.exponential(0.25)
            turbidity = np.random.normal(45.0, 10.0)

        # Inject controlled label noise
        if np.random.rand() < noise_rate:
            ph += np.random.normal(0, 1.0)
            do += np.random.normal(0, 1.5)

        records.append({
            "image_path": row["image_path"],
            "label": row["label"],
            "split": row["split"],
            "ph": round(float(np.clip(ph, 4.0, 10.0)), 2),
            "temperature": round(float(np.clip(temp, 15.0, 40.0)), 2),
            "dissolved_oxygen": round(float(np.clip(do, 0.5, 12.0)), 2),
            "salinity": round(float(np.clip(salinity, 0.0, 40.0)), 2),
            "ammonia": round(float(np.clip(ammonia, 0.0, 3.0)), 3),
            "turbidity": round(float(np.clip(turbidity, 1.0, 100.0)), 2),
        })

    return pd.DataFrame(records)


def train_tabular_model(df: pd.DataFrame, out_dir: Path):
    features = ["ph", "temperature", "dissolved_oxygen", "salinity", "ammonia", "turbidity"]
    X = df[features]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    report = classification_report(y_test, preds, output_dict=True)
    
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(report).transpose().to_csv(out_dir / "tabular_classification_report.csv")
    
    # Feature Importances
    df_imp = pd.DataFrame({"feature": features, "importance": clf.feature_importances_})
    df_imp.to_csv(out_dir / "feature_importances.csv", index=False)

    return clf, report


def run(config):
    manifest_dir = config.paths.datasets / "manifests"
    wq_out_dir = config.paths.datasets / "water_quality"
    wq_out_dir.mkdir(parents=True, exist_ok=True)
    
    out_dir = config.paths.outputs / "water_quality"

    fish_manifest = manifest_dir / "fish_manifest_deduped.csv"
    if fish_manifest.exists():
        df_fish = pd.read_csv(fish_manifest)
        df_fish_wq = generate_synthetic_water_data(df_fish, noise_rate=config.water_quality.label_noise_rate)
        fish_wq_path = wq_out_dir / "fish_water_quality.csv"
        df_fish_wq.to_csv(fish_wq_path, index=False)
        _, report = train_tabular_model(df_fish_wq, out_dir / "fish")
        print(f"Fish Water Quality Dataset created ({len(df_fish_wq)} rows). Macro F1: {report['macro avg']['f1-score']:.4f}")

    shrimp_manifest = manifest_dir / "shrimp_manifest_deduped.csv"
    if shrimp_manifest.exists():
        df_shrimp = pd.read_csv(shrimp_manifest)
        df_shrimp_wq = generate_synthetic_water_data(df_shrimp, noise_rate=config.water_quality.label_noise_rate)
        shrimp_wq_path = wq_out_dir / "shrimp_water_quality.csv"
        df_shrimp_wq.to_csv(shrimp_wq_path, index=False)
        _, report = train_tabular_model(df_shrimp_wq, out_dir / "shrimp")
        print(f"Shrimp Water Quality Dataset created ({len(df_shrimp_wq)} rows). Macro F1: {report['macro avg']['f1-score']:.4f}")


if __name__ == "__main__":
    from config import load_config
    cfg = load_config()
    run(cfg)
