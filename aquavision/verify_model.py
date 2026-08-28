import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import classification_report, accuracy_score, f1_score

from aquavision.preprocessing import letterbox_resize
from aquavision.fusion import AquaVisionMultimodalModel

def run_heldout_test_evaluation(species: str = "fish", fold: int = 0):
    print(f"\n==================================================")
    print(f" EVALUATING HELDOUT TEST SET ({species.upper()} - FOLD {fold} MODEL)")
    print(f"==================================================")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest_path = Path("datasets/processed") / f"{species}_processed_manifest.csv"
    df = pd.read_csv(manifest_path)

    test_df = df[df["fold"] == -1].reset_index(drop=True)
    if len(test_df) == 0:
        print(f"No held-out test samples (fold == -1) found for {species}.")
        return

    classes = sorted(df["label"].unique())
    label2idx = {lbl: idx for idx, lbl in enumerate(classes)}
    test_df["label_idx"] = test_df["label"].map(label2idx)

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
        test_df[tab_cols] = scaler.transform(test_df[tab_cols].fillna(0.0))

    y_true, y_pred = [], []

    with torch.no_grad():
        for idx, row in test_df.iterrows():
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

            y_true.append(row["label_idx"])
            y_pred.append(pred_idx)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    print(f"Total Held-Out Test Samples: {len(test_df)}")
    print(f"Test Accuracy : {acc*100:.2f}%")
    print(f"Test Macro F1 : {macro_f1:.4f}\n")
    print(classification_report(y_true, y_pred, target_names=classes, zero_division=0))

if __name__ == "__main__":
    run_heldout_test_evaluation(species="fish", fold=0)
