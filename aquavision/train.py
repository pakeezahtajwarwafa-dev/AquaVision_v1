import os
import argparse
import joblib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import cv2
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import albumentations as A
from albumentations.pytorch import ToTensorV2

from aquavision.preprocessing import letterbox_resize
from aquavision.fusion import AquaVisionMultimodalModel
from config import load_config


class AquaMultimodalDataset(Dataset):
    def __init__(self, df, img_dir, tab_cols=None, transform=None, species="fish"):
        self.df = df.reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.tab_cols = tab_cols or []
        self.transform = transform
        self.species = species
        self.label_map = {lbl: idx for idx, lbl in enumerate(sorted(self.df["label"].unique()))}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.img_dir / row["image_path"]

        img = cv2.imread(str(img_path))
        if img is None:
            img = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_resized = letterbox_resize(img, target_size=224, species=self.species)

        if self.transform:
            augmented = self.transform(image=img_resized)
            img_tensor = augmented["image"]
        else:
            img_tensor = ToTensorV2()(image=img_resized)["image"]

        label_idx = self.label_map[row["label"]]

        if self.tab_cols:
            tab_values = row[self.tab_cols].values.astype(np.float32)
            return img_tensor, torch.tensor(tab_values, dtype=torch.float32), torch.tensor(label_idx, dtype=torch.long)

        return img_tensor, torch.tensor(label_idx, dtype=torch.long)


def train_cross_validation(species="fish", backbone_name="resnet18", epochs=25, patience=5, min_epochs=10, batch_size=16, lr=3e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n==================================================")
    print(f" Starting Training: Species={species.upper()} | Backbone={backbone_name} | Device={device}")
    print(f"==================================================")

    processed_dir = Path("datasets/processed")
    manifest_path = processed_dir / f"{species}_processed_manifest.csv"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    df = pd.read_csv(manifest_path)
    img_dir = processed_dir / species

    tab_cols = [c for c in ["temperature", "ph", "dissolved_oxygen", "ammonia", "temp_do_ratio", "ammonia_toxicity_index", "stress_score"] if c in df.columns]
    num_classes = len(df["label"].unique())

    # CRITICAL: strictly isolate train/val data from the held-out test set.
    # fold == -1 marks rows carved out by rebuild_manifests.py as the
    # zero-leakage evaluation set -- they must never appear in training or
    # validation for ANY fold.
    cv_df = df[df["fold"] >= 0].copy()
    folds = sorted(cv_df["fold"].unique())

    transform_train = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    transform_val = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    checkpoints_dir = Path("checkpoints")
    checkpoints_dir.mkdir(exist_ok=True)

    for fold in folds:
        print(f"\n--- Fold {fold} ({species} - {backbone_name}) ---")
        train_df = cv_df[cv_df["fold"] != fold].copy()
        val_df = cv_df[cv_df["fold"] == fold].copy()

        scaler = None
        if tab_cols:
            scaler = StandardScaler()
            train_df[tab_cols] = scaler.fit_transform(train_df[tab_cols])
            val_df[tab_cols] = scaler.transform(val_df[tab_cols])

            scaler_path = checkpoints_dir / f"{species}_scaler_fold{fold}.pkl"
            joblib.dump(scaler, scaler_path)

        train_ds = AquaMultimodalDataset(train_df, img_dir, tab_cols, transform=transform_train, species=species)
        val_ds = AquaMultimodalDataset(val_df, img_dir, tab_cols, transform=transform_val, species=species)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

        model = AquaVisionMultimodalModel(
            num_classes=num_classes,
            num_tabular_features=len(tab_cols),
            backbone_name=backbone_name,
            pretrained=True
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_val_acc = 0.0
        patience_counter = 0
        ckpt_path = checkpoints_dir / f"{species}_{backbone_name}_best_model_fold{fold}.pt"

        for epoch in range(epochs):
            model.train()
            train_loss, train_correct, train_total = 0.0, 0, 0

            for batch in train_loader:
                if tab_cols:
                    imgs, tabs, labels = batch
                    imgs, tabs, labels = imgs.to(device), tabs.to(device), labels.to(device)
                    outputs = model(imgs, tabs)
                else:
                    imgs, labels = batch
                    imgs, labels = imgs.to(device), labels.to(device)
                    outputs = model(imgs)

                optimizer.zero_grad()
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * len(labels)
                train_correct += (outputs.argmax(1) == labels).sum().item()
                train_total += len(labels)

            model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            with torch.no_grad():
                for batch in val_loader:
                    if tab_cols:
                        imgs, tabs, labels = batch
                        imgs, tabs, labels = imgs.to(device), tabs.to(device), labels.to(device)
                        outputs = model(imgs, tabs)
                    else:
                        imgs, labels = batch
                        imgs, labels = imgs.to(device), labels.to(device)
                        outputs = model(imgs)

                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * len(labels)
                    val_correct += (outputs.argmax(1) == labels).sum().item()
                    val_total += len(labels)

            train_acc = train_correct / train_total * 100
            val_acc = val_correct / val_total * 100
            print(f"Epoch {epoch+1:02d}/{epochs:02d} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | Val Loss: {val_loss/val_total:.4f}")

            scheduler.step()

            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save(model.state_dict(), ckpt_path)
            else:
                patience_counter += 1

            if epoch + 1 >= min_epochs and patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1} (no improvement for {patience} epochs, min_epochs={min_epochs} satisfied).")
                break

        print(f"Fold {fold} Best Val Accuracy: {best_val_acc:.2f}% (Saved to {ckpt_path.name})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AquaVision multi-backbone models.")
    parser.add_argument("--species", type=str, default="fish", choices=["fish", "shrimp"], help="Target species")
    parser.add_argument("--backbones", nargs="+", default=["resnet18"], help="List of backbones to train (e.g. resnet18 efficientnet_b0 vit_b_16)")
    parser.add_argument("--epochs", type=int, default=25, help="Max training epochs per fold (early stopping may end sooner)")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience (epochs without val improvement)")
    parser.add_argument("--min-epochs", type=int, default=10, help="Minimum epochs before early stopping can trigger")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    args = parser.parse_args()

    for backbone in args.backbones:
        train_cross_validation(
            species=args.species,
            backbone_name=backbone,
            epochs=args.epochs,
            patience=args.patience,
            min_epochs=args.min_epochs,
            batch_size=args.batch_size,
            lr=args.lr
        )
