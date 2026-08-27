import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

from aquavision.dataset import AquaDataset
from aquavision.fusion import AquaVisionMultimodalModel

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        optimizer.zero_grad()
        if len(batch) == 3:
            images, tabular, labels = batch
            images, tabular, labels = images.to(device), tabular.to(device), labels.to(device)
            outputs = model(images, tabular)
        else:
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(dataloader.dataset)

def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
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

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    return acc, f1

def run_cross_validation(species: str = "fish", epochs: int = 15, batch_size: int = 16, lr: float = 3e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- Running 5-Fold Augmented Cross-Validation: {species.upper()} ---")

    manifest_path = Path("datasets/processed") / f"{species}_processed_manifest.csv"
    if not manifest_path.exists():
        print(f"Manifest missing for {species}")
        return

    df = pd.read_csv(manifest_path)
    unique_labels = sorted(df["label"].unique())
    label2idx = {lbl: idx for idx, lbl in enumerate(unique_labels)}
    df["label_idx"] = df["label"].map(label2idx)

    tab_cols = [c for c in ["temperature", "ph", "dissolved_oxygen", "ammonia", "temp_do_ratio", "ammonia_toxicity_index", "stress_score"] if c in df.columns]
    
    # Save overall scaler for deployment/inference consistency
    checkpoints_dir = Path("checkpoints")
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    if tab_cols:
        full_scaler = StandardScaler()
        full_scaler.fit(df[tab_cols].fillna(0.0))
        joblib.dump(full_scaler, checkpoints_dir / f"{species}_scaler.pkl")

    fold_f1s = []

    for fold in range(5):
        train_df = df[df["fold"] != fold].copy()
        val_df = df[df["fold"] == fold].copy()

        if tab_cols:
            scaler = StandardScaler()
            train_df[tab_cols] = scaler.fit_transform(train_df[tab_cols].fillna(0.0))
            val_df[tab_cols] = scaler.transform(val_df[tab_cols].fillna(0.0))

        train_ds = AquaDataset(train_df, is_train=True, species=species, tab_cols=tab_cols)
        val_ds = AquaDataset(val_df, is_train=False, species=species, tab_cols=tab_cols)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        model = AquaVisionMultimodalModel(
            num_classes=len(unique_labels),
            num_tabular_features=len(tab_cols),
            backbone_name="resnet18"
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_f1, best_acc = 0.0, 0.0
        best_checkpoint = checkpoints_dir / f"{species}_best_model_fold{fold}.pt"

        for epoch in range(epochs):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_acc, val_f1 = evaluate(model, val_loader, device)
            scheduler.step()

            if val_f1 > best_f1:
                best_f1 = val_f1
                best_acc = val_acc
                torch.save(model.state_dict(), best_checkpoint)

        fold_f1s.append(best_f1)
        print(f"  > Fold {fold} -> Best Val Acc: {best_acc:.4f} | Best Macro F1: {best_f1:.4f}")

    mean_f1 = float(np.mean(fold_f1s))
    var_f1 = float(np.var(fold_f1s))
    std_f1 = float(np.std(fold_f1s))

    print(f"\n[{species.upper()} STATISTICAL SUMMARY]")
    print(f"  - Fold F1 Scores : {[round(x, 4) for x in fold_f1s]}")
    print(f"  - Mean Macro F1   : {mean_f1:.4f}")
    print(f"  - F1 Variance     : {var_f1:.6f}")
    print(f"  - F1 Std Dev      : {std_f1:.4f}")
    print(f"  - Stability Check : {'STABLE' if std_f1 < 0.03 else 'UNSTABLE'}")

def main():
    print("="*50)
    print(" 5-FOLD CV WITH ONLINE AUGMENTATIONS & COSINE SCHEDULER")
    print("="*50)
    for species in ["fish", "shrimp"]:
        run_cross_validation(species=species, epochs=15)

if __name__ == "__main__":
    main()
