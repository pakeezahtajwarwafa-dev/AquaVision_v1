import torch
import pandas as pd
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

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

def evaluate_model(species="fish", backbone="efficientnet_b0", best_fold=3):
    csv_path = Path(f"datasets/processed/{species}_processed_manifest.csv")
    df = pd.read_csv(csv_path)
    
    # 1. Rebuild Class Mappings from Training Data
    train_df = df[df['fold'] >= 0]
    classes = sorted(train_df['label'].unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    num_classes = len(classes)
    
    # 2. Isolate the Zero-Leakage Test Set
    test_df = df[df['fold'] == -1].copy()
    print(f"\n--- Zero-Leakage Evaluation: {species.upper()} ---")
    print(f"Evaluating Fold {best_fold} on {len(test_df)} held-out samples...")

    if len(test_df) == 0:
        print("[ERROR] No held-out test samples found (fold == -1).")
        return

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_ds = AquaDataset(test_df, transform=test_transform, class_to_idx=class_to_idx)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 3. Load the Best Model Weights
    model = AquaVisionModel(num_classes=num_classes, backbone=backbone).to(device)
    ckpt_path = Path(f"checkpoints/{backbone}_{species}_fold{best_fold}_best.pt")
    
    if not ckpt_path.exists():
        print(f"[ERROR] Could not find checkpoint: {ckpt_path}")
        return
        
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    all_preds = []
    all_labels = []

    # 4. Run Inference
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    # 5. Generate Reports
    print("\n[ Classification Report ]")
    print(classification_report(all_labels, all_preds, target_names=classes, zero_division=0))

    print("\n[ Confusion Matrix ]")
    cm = confusion_matrix(all_labels, all_preds)

    # Print a nicely formatted text-based confusion matrix
    header = f"{'True / Pred':<35} " + " ".join([f"{i:>5}" for i in range(num_classes)])
    print(header)
    print("-" * len(header))
    for i, class_name in enumerate(classes):
        row_str = f"{i}. {class_name[:30]:<31} | " + " ".join([f"{cm[i][j]:>5}" for j in range(num_classes)])
        print(row_str)

if __name__ == "__main__":
    # Pointing explicitly to fold 3 based on your training logs
    evaluate_model(species="fish", backbone="efficientnet_b0", best_fold=3)
