import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import pandas as pd
from pathlib import Path
from PIL import Image

class CombinedSpeciesDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row['image_path']).convert('RGB')
        label = row['species_label'] # 0 for fish, 1 for shrimp
        if self.transform:
            image = self.transform(image)
        return image, label

def train_gate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training Species Gate on device: {device}")

    # Load clean manifests
    fish_df = pd.read_csv('datasets/processed/fish_processed_manifest.csv')
    shrimp_df = pd.read_csv('datasets/processed/shrimp_processed_manifest.csv')

    fish_df['species_label'] = 0
    shrimp_df['species_label'] = 1

    # Combine training folds
    train_fish = fish_df[fish_df['fold'] != -1]
    train_shrimp = shrimp_df[shrimp_df['fold'] != -1]
    combined_df = pd.concat([train_fish, train_shrimp], ignore_index=True)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = CombinedSpeciesDataset(combined_df, transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Lightweight backbone: ResNet18
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    model.train()
    for epoch in range(3): # 3 epochs is plenty for a simple 2-class problem
        total_loss = 0.0
        correct = 0
        total = 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        acc = correct / total
        print(f"Gate Epoch {epoch+1}/3 - Loss: {total_loss/total:.4f} - Accuracy: {acc*100:.2f}%")

    Path("checkpoints").mkdir(exist_ok=True)
    out_path = Path("checkpoints/species_gate_best.pt")
    torch.save(model.state_dict(), out_path)
    print(f"Species Gate saved to {out_path}")

if __name__ == '__main__':
    train_gate()
