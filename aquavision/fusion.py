import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import cv2
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score

from aquavision.dataset import get_transforms
from aquavision.models import build_model


class MultimodalAquaDataset(Dataset):
    """Loads images alongside tabular water quality features."""
    def __init__(self, df, label_to_idx=None, transform=None, scaler=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        
        self.tab_cols = ["ph", "temperature", "dissolved_oxygen", "salinity", "ammonia", "turbidity"]
        raw_tabular = self.df[self.tab_cols].values.astype(np.float32)
        
        if scaler is None:
            self.scaler = StandardScaler()
            self.tabular_data = self.scaler.fit_transform(raw_tabular)
        else:
            self.scaler = scaler
            self.tabular_data = self.scaler.transform(raw_tabular)

        if label_to_idx is None:
            unique_labels = sorted(self.df['label'].unique())
            self.label_to_idx = {lbl: i for i, lbl in enumerate(unique_labels)}
        else:
            self.label_to_idx = label_to_idx
            
        self.idx_to_label = {v: k for k, v in self.label_to_idx.items()}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = cv2.imread(row['image_path'])
        if image is None:
            raise FileNotFoundError(f"Failed to load: {row['image_path']}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image=image)['image']

        tabular_vec = torch.tensor(self.tabular_data[idx], dtype=torch.float32)
        label_idx = torch.tensor(self.label_to_idx[row['label']], dtype=torch.long)

        return image, tabular_vec, label_idx


class FusionModel(nn.Module):
    """Late-fusion architecture combining CNN visual features with tabular water quality inputs."""
    def __init__(self, backbone_name="resnet34", num_classes=8, pretrained=True, tabular_dim=6):
        super().__init__()
        base = build_model(backbone_name, num_classes=num_classes, pretrained=pretrained)
        
        # Strip final classification head to use model as feature extractor
        self.visual_encoder = nn.Sequential(*list(base.children())[:-1])
        num_visual_features = base.get_classifier().in_features

        self.fusion_head = nn.Sequential(
            nn.Linear(num_visual_features + tabular_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, img, tab):
        feat = self.visual_encoder(img)
        feat = torch.flatten(feat, 1)
        fused = torch.cat([feat, tab], dim=1)
        return self.fusion_head(fused)


def run_fusion_experiment(config, dataset_type="fish", epochs=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wq_csv = config.paths.datasets / "water_quality" / f"{dataset_type}_water_quality.csv"

    if not wq_csv.exists():
        raise FileNotFoundError(f"Water quality dataset missing: {wq_csv}")

    df = pd.read_csv(wq_csv)
    df_train = df[df["split"] == "train"].reset_index(drop=True)
    df_val = df[df["split"] == "val"].reset_index(drop=True)

    train_tf = get_transforms(img_size=config.training.img_size, is_train=True)
    val_tf = get_transforms(img_size=config.training.img_size, is_train=False)

    train_ds = MultimodalAquaDataset(df_train, transform=train_tf)
    val_ds = MultimodalAquaDataset(df_val, label_to_idx=train_ds.label_to_idx, transform=val_tf, scaler=train_ds.scaler)

    train_loader = DataLoader(train_ds, batch_size=config.training.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.training.batch_size, shuffle=False)

    num_classes = len(train_ds.label_to_idx)
    model = FusionModel("resnet34", num_classes=num_classes, pretrained=True).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.lr)

    print(f"\n--- Starting Multimodal Fusion Experiment ({dataset_type.upper()} | Device: {device}) ---")
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for imgs, tabs, targets in train_loader:
            imgs, tabs, targets = imgs.to(device), tabs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(imgs, tabs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)

        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for imgs, tabs, targets in val_loader:
                imgs, tabs, targets = imgs.to(device), tabs.to(device), targets.to(device)
                outputs = model(imgs, tabs)
                all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        val_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss/len(train_ds):.4f} | Val Macro F1: {val_f1:.4f}")

    save_dir = config.paths.outputs / "experiments" / f"{dataset_type}_multimodal_fusion"
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), save_dir / "best_fusion_model.pth")
    print(f"Fusion model saved to: {save_dir / 'best_fusion_model.pth'}\n")


if __name__ == "__main__":
    from config import load_config
    cfg = load_config()
    run_fusion_experiment(cfg, dataset_type="fish", epochs=3)
