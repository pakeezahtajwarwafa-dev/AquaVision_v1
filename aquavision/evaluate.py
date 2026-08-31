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
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import albumentations as A
from albumentations.pytorch import ToTensorV2

from aquavision.preprocessing import letterbox_resize
from aquavision.fusion import AquaVisionMultimodalModel


class AquaEvalDataset(Dataset):
    def __init__(self, df, img_dir, tab_cols=None, transform=None, species="fish", label_map=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.tab_cols = tab_cols or []
        self.transform = transform
        self.species = species
        self.label_map = label_map or {lbl: idx for idx, lbl in enumerate(sorted(self.df["label"].unique()))}

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

        label_idx = self.label_map.get(row["label"], 0)

        if self.tab_cols:
            tab_values = row[self.tab_cols].values.astype(np.float32)
            return img_tensor, torch.tensor(tab_values, dtype=torch.float32), torch.tensor(label_idx, dtype=torch.long)

        return img_tensor, torch.tensor(label_idx, dtype=torch.long)


def evaluate_model(species="fish", backbone_name="resnet18", fold=0, checkpoint_prefix=None, use_test_split=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if checkpoint_prefix is None:
        checkpoint_prefix = f"{species}_{backbone_name}_best_model"

    processed_dir = Path("datasets/processed")
    manifest_path = processed_dir / f"{species}_processed_manifest.csv"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    df = pd.read_csv(manifest_path)
    img_dir = processed_dir / species
    classes = sorted(df["label"].unique())
    label_map = {lbl: idx for idx, lbl in enumerate(classes)}

    tab_cols = [c for c in ["temperature", "ph", "dissolved_oxygen", "ammonia", "temp_do_ratio", "ammonia_toxicity_index", "stress_score"] if c in df.columns]

    # Select evaluation split (Holdout test split if available, otherwise fold validation split)
    if use_test_split and "split" in df.columns and (df["split"] == "test").any():
        eval_df = df[df["split"] == "test"].copy()
        split_desc = "Holdout Test Set"
    elif "fold" in df.columns:
        eval_df = df[df["fold"] == fold].copy()
        split_desc = f"Fold {fold} Validation Set"
    else:
        eval_df = df.copy()
        split_desc = "Full Dataset"

    print(f"\n==================================================")
    print(f" Evaluating: Species={species.upper()} | Backbone={backbone_name} | Split={split_desc}")
    print(f" Target Checkpoint Prefix: {checkpoint_prefix}")
    print(f"==================================================")

    scaler_path = Path("checkpoints") / f"{species}_scaler_fold{fold}.pkl"
    if tab_cols and scaler_path.exists():
        scaler = joblib.load(scaler_path)
        eval_df[tab_cols] = scaler.transform(eval_df[tab_cols])

    transform_val = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    eval_ds = AquaEvalDataset(eval_df, img_dir, tab_cols, transform=transform_val, species=species, label_map=label_map)
    eval_loader = DataLoader(eval_ds, batch_size=16, shuffle=False, num_workers=0)

    ckpt_path = Path("checkpoints") / f"{checkpoint_prefix}_fold{fold}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint model weights not found at: {ckpt_path}")

    model = AquaVisionMultimodalModel(
        num_classes=len(classes),
        num_tabular_features=len(tab_cols),
        backbone_name=backbone_name,
        pretrained=False
    ).to(device)

    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in eval_loader:
            if tab_cols:
                imgs, tabs, labels = batch
                imgs, tabs = imgs.to(device), tabs.to(device)
                outputs = model(imgs, tabs)
            else:
                imgs, labels = batch
                imgs = imgs.to(device)
                outputs = model(imgs)

            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(labels.numpy())

    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)

    print(f"\n--- Performance Summary ---")
    print(f"Accuracy: {acc*100:.2f}%")
    print(f"Macro F1: {macro_f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(all_targets, all_preds, target_names=classes, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(all_targets, all_preds))

    return {"accuracy": acc, "macro_f1": macro_f1}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate AquaVision models.")
    parser.add_argument("--species", type=str, default="fish", choices=["fish", "shrimp"], help="Target species")
    parser.add_argument("--backbone", type=str, default="resnet18", help="Backbone architecture (e.g. resnet18, efficientnet_b0)")
    parser.add_argument("--fold", type=int, default=0, help="Fold index")
    parser.add_argument("--checkpoint-prefix", type=str, default=None, help="Custom checkpoint prefix override")
    args = parser.parse_args()

    evaluate_model(
        species=args.species,
        backbone_name=args.backbone,
        fold=args.fold,
        checkpoint_prefix=args.checkpoint-prefix
    )
