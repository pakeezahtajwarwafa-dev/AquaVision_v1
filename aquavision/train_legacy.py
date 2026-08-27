import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

from aquavision.dataset import AquaDataset
from aquavision.fusion import AquaVisionMultimodalModel

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for batch in loader:
        if len(batch) == 3:
            images, tabular, labels = batch
            images, tabular, labels = images.to(device), tabular.to(device), labels.to(device)
            outputs = model(images, tabular)
        else:
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average="macro")
    return epoch_loss, epoch_acc, epoch_f1

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                images, tabular, labels = batch
                images, tabular, labels = images.to(device), tabular.to(device), labels.to(device)
                outputs = model(images, tabular)
            else:
                images, labels = batch
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)

            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss = running_loss / len(loader.dataset)
    val_acc = accuracy_score(all_labels, all_preds)
    val_f1 = f1_score(all_labels, all_preds, average="macro")
    return val_loss, val_acc, val_f1

def run_fold(species: str, fold_to_run: int, epochs: int = 5, batch_size: int = 16, lr: float = 1e-4, device=None):
    manifest_path = Path("datasets/processed") / f"{species}_processed_manifest.csv"
    df = pd.read_csv(manifest_path)

    unique_labels = sorted(df["label"].unique())
    label2idx = {lbl: idx for idx, lbl in enumerate(unique_labels)}
    df["label_idx"] = df["label"].map(label2idx)

    tab_cols = [c for c in ["temperature", "ph", "dissolved_oxygen", "ammonia", "temp_do_ratio", "ammonia_toxicity_index", "stress_score"] if c in df.columns]

    train_df = df[df["fold"] != fold_to_run].copy()
    val_df = df[df["fold"] == fold_to_run].copy()

    if tab_cols:
        scaler = StandardScaler()
        train_df[tab_cols] = scaler.fit_transform(train_df[tab_cols].fillna(0.0))
        val_df[tab_cols] = scaler.transform(val_df[tab_cols].fillna(0.0))

    train_ds = AquaDataset(train_df, img_size=224, is_train=True, species=species, tab_cols=tab_cols)
    val_ds = AquaDataset(val_df, img_size=224, is_train=False, species=species, tab_cols=tab_cols)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = AquaVisionMultimodalModel(
        num_classes=len(unique_labels),
        num_tabular_features=len(tab_cols),
        backbone_name="resnet18"
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    checkpoints_dir = Path("checkpoints")
    checkpoints_dir.mkdir(exist_ok=True)

    best_val_f1 = 0.0
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc, tr_f1 = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_acc = val_acc
            save_path = checkpoints_dir / f"{species}_best_model_fold{fold_to_run}.pt"
            torch.save(model.state_dict(), save_path)

    print(f"  > Fold {fold_to_run} -> Best Val Acc: {best_val_acc:.4f} | Best Macro F1: {best_val_f1:.4f}")
    return best_val_acc, best_val_f1

def run_full_cross_validation(epochs: int = 5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"==================================================")
    print(f" 5-FOLD CV & VARIANCE ANALYSIS | Device: {device}")
    print(f"==================================================\n")
    
    summary_results = []
    
    for species in ["fish", "shrimp"]:
        manifest_path = Path("datasets/processed") / f"{species}_processed_manifest.csv"
        if not manifest_path.exists():
            continue
            
        print(f"--- Running 5-Fold Cross-Validation: {species.upper()} ---")
        fold_accs, fold_f1s = [], []
        
        for fold in range(5):
            acc, f1 = run_fold(species=species, fold_to_run=fold, epochs=epochs, device=device)
            fold_accs.append(acc)
            fold_f1s.append(f1)
            
        mean_acc = float(np.mean(fold_accs))
        mean_f1 = float(np.mean(fold_f1s))
        var_f1 = float(np.var(fold_f1s))
        std_f1 = float(np.std(fold_f1s))
        
        # Stability assessment threshold
        status = "STABLE" if std_f1 <= 0.03 else "HIGH VARIANCE (Investigate)"

        print(f"\n[{species.upper()} STATISTICAL SUMMARY]")
        print(f"  - Fold F1 Scores : {[round(f, 4) for f in fold_f1s]}")
        print(f"  - Mean Macro F1   : {mean_f1:.4f}")
        print(f"  - F1 Variance     : {var_f1:.6f}")
        print(f"  - F1 Std Dev      : {std_f1:.4f}")
        print(f"  - Stability Check : {status}")
        print("\n" + "="*50 + "\n")
        
        summary_results.append({
            "species": species,
            "mean_accuracy": round(mean_acc, 4),
            "mean_macro_f1": round(mean_f1, 4),
            "f1_variance": round(var_f1, 6),
            "f1_std_dev": round(std_f1, 4),
            "status": status
        })

    df_res = pd.DataFrame(summary_results)
    df_res.to_csv("datasets/cv_metrics.csv", index=False)
    print("Saved 5-Fold CV analysis to datasets/cv_metrics.csv")

if __name__ == "__main__":
    run_full_cross_validation(epochs=5)
