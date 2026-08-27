import torch
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

from aquavision.dataset import AquaDataset
from aquavision.fusion import AquaVisionMultimodalModel

def get_confusion_matrix_df(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred)
    return pd.DataFrame(cm, index=labels, columns=labels)

def evaluate_test_set(species: str = "fish", fold_to_eval: int = 0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== HELD-OUT EVALUATION: {species.upper()} (Fold {fold_to_eval}) ===")

    processed_manifest = Path("datasets/processed") / f"{species}_processed_manifest.csv"
    if not processed_manifest.exists():
        return

    df = pd.read_csv(processed_manifest)
    
    # Filter for validation fold
    test_df = df[df["fold"] == fold_to_eval].copy()

    unique_labels = sorted(df["label"].unique())
    label2idx = {lbl: idx for idx, lbl in enumerate(unique_labels)}
    test_df["label_idx"] = test_df["label"].map(label2idx)

    tab_cols = [c for c in ["temperature", "ph", "dissolved_oxygen", "ammonia", "temp_do_ratio", "ammonia_toxicity_index", "stress_score"] if c in df.columns]

    if tab_cols:
        scaler = StandardScaler()
        train_df = df[df["fold"] != fold_to_eval]
        scaler.fit(train_df[tab_cols].fillna(0.0))
        test_df[tab_cols] = scaler.transform(test_df[tab_cols].fillna(0.0))

    test_ds = AquaDataset(test_df, img_size=224, is_train=False, species=species, tab_cols=tab_cols)
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

    checkpoint_path = Path("checkpoints") / f"{species}_best_model_fold{fold_to_eval}.pt"
    if not checkpoint_path.exists():
        print(f"  > Checkpoint not found: {checkpoint_path}")
        return

    model = AquaVisionMultimodalModel(
        num_classes=len(unique_labels),
        num_tabular_features=len(tab_cols),
        backbone_name="resnet18"
    ).to(device)

    # Safe weights loading with weights_only=True
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 3:
                images, tabular, labels = batch
                images, tabular = images.to(device), tabular.to(device)
                outputs = model(images, tabular)
            else:
                images, labels = batch
                images = images.to(device)
                outputs = model(images)

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    print("\nClassification Report:")
    report = classification_report(all_labels, all_preds, target_names=unique_labels, digits=4)
    print(report)

    cm_df = get_confusion_matrix_df(all_labels, all_preds, unique_labels)
    cm_out = Path("datasets") / f"{species}_confusion_matrix.csv"
    cm_df.to_csv(cm_out)
    print(f"Saved confusion matrix to: {cm_out}\n" + "="*50 + "\n")

def run():
    for species in ["fish", "shrimp"]:
        evaluate_test_set(species)

if __name__ == "__main__":
    run()
