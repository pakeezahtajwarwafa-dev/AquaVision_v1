import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

from aquavision.preprocessing import letterbox_resize
from aquavision.fusion import AquaVisionMultimodalModel

def run_tabular_sensitivity_test(species: str = "fish", fold: int = 0):
    print(f"\n==================================================")
    print(f" TEST 1: TABULAR SENSITIVITY TEST ({species.upper()} - FOLD {fold})")
    print(f"==================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest_path = Path("datasets/processed") / f"{species}_processed_manifest.csv"
    df = pd.read_csv(manifest_path)

    classes = sorted(df["label"].unique())
    tab_cols = [c for c in ["temperature", "ph", "dissolved_oxygen", "ammonia", "temp_do_ratio", "ammonia_toxicity_index", "stress_score"] if c in df.columns]

    # Load fold-matched scaler
    scaler_path = Path("checkpoints") / f"{species}_scaler_fold{fold}.pkl"
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None

    checkpoint_path = Path("checkpoints") / f"{species}_best_model_fold{fold}.pt"
    model = AquaVisionMultimodalModel(num_classes=len(classes), num_tabular_features=len(tab_cols), backbone_name="resnet18").to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()

    transform = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    sample_row = df.iloc[0]
    img = cv2.imread(sample_row["image_path"])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = letterbox_resize(img, target_size=224, species=species)
    img_tensor = transform(image=img)["image"].unsqueeze(0).to(device)

    water_scenarios = {
        "Optimal Water Quality": {"temperature": 25.0, "ph": 7.5, "dissolved_oxygen": 7.5, "ammonia": 0.01},
        "Severe Water Stress" : {"temperature": 32.0, "ph": 8.8, "dissolved_oxygen": 2.1, "ammonia": 1.20}
    }

    print(f"Sample Image: {sample_row['image_path']}")
    print(f"Ground Truth: {sample_row['label']}\n")

    for env_name, wq in water_scenarios.items():
        wq_dict = wq.copy()
        wq_dict["temp_do_ratio"] = wq_dict["temperature"] / (wq_dict["dissolved_oxygen"] + 1e-5)
        wq_dict["ammonia_toxicity_index"] = wq_dict["ammonia"] * (10 ** (wq_dict["ph"] - 7.0)) * (wq_dict["temperature"] / 25.0)
        wq_dict["stress_score"] = (abs(wq_dict["ph"] - 7.5) * 0.3) + (wq_dict["ammonia"] * 10.0 * 0.4) + (max(0, 6.0 - wq_dict["dissolved_oxygen"]) * 0.3)

        raw_df = pd.DataFrame([[wq_dict[c] for c in tab_cols]], columns=tab_cols)
        scaled_vector = scaler.transform(raw_df)[0] if scaler else raw_df.values[0]

        tab_tensor = torch.tensor(scaled_vector, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(img_tensor, tab_tensor)
            probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()

        top_idx = int(np.argmax(probs))
        print(f"[{env_name}] -> Stress Score: {wq_dict['stress_score']:.2f}")
        print(f"   Predicted Class : {classes[top_idx]}")
        print(f"   Confidence      : {probs[top_idx]*100:.2f}%\n")


def run_error_audit(species: str = "fish", fold: int = 0, max_errors: int = 5):
    print(f"==================================================")
    print(f" TEST 2: MISCLASSIFICATION AUDIT ({species.upper()} - FOLD {fold})")
    print(f"==================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest_path = Path("datasets/processed") / f"{species}_processed_manifest.csv"
    df = pd.read_csv(manifest_path)

    val_df = df[df["fold"] == fold].reset_index(drop=True)
    classes = sorted(df["label"].unique())
    label2idx = {lbl: idx for idx, lbl in enumerate(classes)}
    val_df["label_idx"] = val_df["label"].map(label2idx)

    tab_cols = [c for c in ["temperature", "ph", "dissolved_oxygen", "ammonia", "temp_do_ratio", "ammonia_toxicity_index", "stress_score"] if c in df.columns]

    scaler_path = Path("checkpoints") / f"{species}_scaler_fold{fold}.pkl"
    scaler = joblib.load(scaler_path) if scaler_path.exists() else None

    checkpoint_path = Path("checkpoints") / f"{species}_best_model_fold{fold}.pt"
    model = AquaVisionMultimodalModel(num_classes=len(classes), num_tabular_features=len(tab_cols), backbone_name="resnet18").to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()

    transform = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    if tab_cols and scaler:
        val_df[tab_cols] = scaler.transform(val_df[tab_cols].fillna(0.0))

    errors = []
    with torch.no_grad():
        for idx, row in val_df.iterrows():
            img = cv2.imread(row["image_path"])
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = letterbox_resize(img, target_size=224, species=species)
            img_tensor = transform(image=img)["image"].unsqueeze(0).to(device)

            if tab_cols:
                tab_vector = np.array([float(row[c]) for c in tab_cols], dtype=np.float32)
                tab_tensor = torch.tensor(tab_vector, dtype=torch.float32).unsqueeze(0).to(device)
                logits = model(img_tensor, tab_tensor)
            else:
                logits = model(img_tensor)

            probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()
            pred_idx = int(np.argmax(probs))

            if pred_idx != row["label_idx"]:
                errors.append({
                    "path": row["image_path"],
                    "true": row["label"],
                    "pred": classes[pred_idx],
                    "conf": probs[pred_idx] * 100
                })

    acc = (1 - len(errors) / len(val_df)) * 100
    print(f"Total Validation Samples: {len(val_df)}")
    print(f"Total Misclassifications: {len(errors)} (Accuracy: {acc:.2f}%)\n")

    print(f"Top {min(max_errors, len(errors))} Error Case Samples:")
    for i, err in enumerate(errors[:max_errors]):
        print(f" [{i+1}] File: {err['path']}")
        print(f"     True Class : {err['true']}")
        print(f"     Pred Class : {err['pred']} ({err['conf']:.2f}% confidence)\n")

if __name__ == "__main__":
    run_tabular_sensitivity_test(species="fish", fold=0)
    run_error_audit(species="fish", fold=0, max_errors=5)
    run_error_audit(species="shrimp", fold=0, max_errors=5)
