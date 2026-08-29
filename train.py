import os
import torch
import pandas as pd
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from aquavision.models.vision_backbone import AquaVisionModel

class AquaDataset(Dataset):
    def __init__(self, df, transform=None, class_to_idx=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.class_to_idx = class_to_idx

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.loc[idx, 'image_path']
        label_str = self.df.loc[idx, 'label']
        
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
            
        label = self.class_to_idx[label_str]
        return image, label

def train_folds(species="fish", backbone="efficientnet_b0", epochs=5, batch_size=32):
    csv_path = Path(f"datasets/processed/{species}_processed_manifest.csv")
    if not csv_path.exists():
        print(f"[ERROR] Could not find {csv_path}")
        return

    df = pd.read_csv(csv_path)
    
    # Exclude the zero-leakage test set (fold -1)
    train_df = df[df['fold'] >= 0].copy()
    
    # Map string labels to integers
    classes = sorted(train_df['label'].unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    num_classes = len(classes)
    
    print(f"\\n--- Starting 5-Fold Training for {species.upper()} ---")
    print(f"Classes ({num_classes}): {classes}")
    
    # Ensure checkpoints directory exists
    Path("checkpoints").mkdir(exist_ok=True)

    # Standard ImageNet transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    for fold in range(5):
        print(f"\\n=== FOLD {fold} ===")
        
        train_sub = train_df[train_df['fold'] != fold]
        val_sub = train_df[train_df['fold'] == fold]
        
        train_ds = AquaDataset(train_sub, transform=train_transform, class_to_idx=class_to_idx)
        val_ds = AquaDataset(val_sub, transform=val_transform, class_to_idx=class_to_idx)
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
        
        model = AquaVisionModel(num_classes=num_classes, backbone=backbone).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        
        best_val_acc = 0.0
        
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            
            for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    
            val_acc = correct / total
            print(f"Fold {fold} | Epoch {epoch+1} | Val Acc: {val_acc:.4f}")
            
            # Save best model for this fold
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                ckpt_name = f"checkpoints/{backbone}_fish_fold{fold}_best.pt"
                torch.save(model.state_dict(), ckpt_name)
                
        print(f"Best Val Accuracy for Fold {fold}: {best_val_acc:.4f}")

if __name__ == "__main__":
    # Feel free to change backbone to 'resnet50' and increase epochs
    train_folds(species="fish", backbone="efficientnet_b0", epochs=5)
