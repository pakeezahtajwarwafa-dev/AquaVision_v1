import os
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score

from aquavision.dataset import AquaDataset, get_transforms
from aquavision.models import build_model


def get_class_weights(df: pd.DataFrame, label_to_idx: dict) -> torch.Tensor:
    """Computes inverse class frequencies to address class imbalance in CrossEntropyLoss."""
    counts = df["label"].map(label_to_idx).value_counts().sort_index()
    total = len(df)
    n_classes = len(label_to_idx)
    weights = total / (n_classes * counts.values)
    return torch.tensor(weights, dtype=torch.float)


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds, all_targets = [], []

    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_targets, all_preds)
    epoch_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    return epoch_loss, epoch_acc, epoch_f1


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_targets = [], []

    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        outputs = model(images)
        loss = criterion(outputs, targets)

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_targets, all_preds)
    epoch_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    return epoch_loss, epoch_acc, epoch_f1


def run_experiment(
    config,
    dataset_type: str = "fish",
    model_name: str = "resnet34",
    exp_name: str = "baseline",
    use_augmentation: bool = True,
    use_pretrained: bool = True,
    use_class_weighting: bool = True,
    epochs: int = None
):
    """Executes a complete training and evaluation run with logging and best-model checkpointing."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = epochs or config.training.epochs

    manifest_path = config.paths.datasets / "manifests" / f"{dataset_type}_manifest_deduped.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    df = pd.read_csv(manifest_path)
    df_train = df[df["split"] == "train"].reset_index(drop=True)
    df_val = df[df["split"] == "val"].reset_index(drop=True)

    train_transform = get_transforms(
        img_size=config.training.img_size, is_train=True, use_augmentation=use_augmentation
    )
    val_transform = get_transforms(img_size=config.training.img_size, is_train=False)

    train_ds = AquaDataset(df_train, transform=train_transform)
    val_ds = AquaDataset(df_val, label_to_idx=train_ds.label_to_idx, transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=config.training.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=config.training.batch_size, shuffle=False, num_workers=0)

    num_classes = len(train_ds.label_to_idx)
    model = build_model(model_name=model_name, num_classes=num_classes, pretrained=use_pretrained).to(device)

    if use_class_weighting:
        weights = get_class_weights(df_train, train_ds.label_to_idx).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.lr)

    save_dir = config.paths.outputs / "experiments" / exp_name
    save_dir.mkdir(parents=True, exist_ok=True)

    best_f1 = 0.0
    history = []

    print(f"\n--- Starting Experiment: {exp_name} ({dataset_type.upper()} | Model: {model_name} | Device: {device}) ---")
    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc, train_f1 = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss, "train_acc": train_acc, "train_f1": train_f1,
            "val_loss": val_loss, "val_acc": val_acc, "val_f1": val_f1,
        })

        print(
            f"Epoch {epoch:02d}/{num_epochs:02d} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} F1: {train_f1:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), save_dir / "best_model.pth")

    pd.DataFrame(history).to_csv(save_dir / "history.csv", index=False)
    print(f"[{exp_name}] Completed. Best Val Macro F1: {best_f1:.4f}\n")
    return save_dir / "best_model.pth", best_f1


if __name__ == "__main__":
    from config import load_config

    cfg = load_config()
    # Smoke test run (1 epoch) to confirm pipeline end-to-end execution
    run_experiment(cfg, dataset_type="fish", model_name="resnet34", exp_name="smoke_test", epochs=1)
