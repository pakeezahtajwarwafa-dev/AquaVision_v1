import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import cv2
import pandas as pd
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2

class SpeciesGateDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(str(path))
        if img is None:
            # Fallback for unreadable images
            img = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224))
        if self.transform:
            img = self.transform(image=img)["image"]
        return img, torch.tensor(label, dtype=torch.long)

def train_gate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training 3-Class Species & OOD Gate on device: {device}")

    samples = []
    class_map = {"fish": 0, "shrimp": 1, "other": 2}
    
    for label_name, label_idx in class_map.items():
        folder = Path("datasets/processed") / label_name
        if label_name in ["fish", "shrimp"]:
            manifest = Path("datasets/processed") / f"{label_name}_processed_manifest.csv"
            if manifest.exists():
                df = pd.read_csv(manifest)
                for _, row in df.iterrows():
                    img_path = folder / row["image_path"]
                    if img_path.exists():
                        samples.append((img_path, label_idx))
        else:
            for img_path in folder.glob("*.jpg"):
                samples.append((img_path, label_idx))

    transform = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    dataset = SpeciesGateDataset(samples, transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)

    # ResNet18 configured for 3 classes
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 3)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    model.train()
    for epoch in range(4):
        total_loss, correct, total = 0.0, 0, 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(labels)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += len(labels)

        print(f"Gate Epoch {epoch+1}/4 - Loss: {total_loss/total:.4f} - Accuracy: {correct/total*100:.2f}%")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/species_gate_best.pt")
    print("Saved 3-Class Gate to checkpoints/species_gate_best.pt")

if __name__ == "__main__":
    train_gate()
