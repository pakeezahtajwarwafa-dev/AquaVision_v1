import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transforms(img_size: int = 224, is_train: bool = True, use_augmentation: bool = True):
    """Returns Albumentations transform pipelines for training and validation."""
    if is_train and use_augmentation:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


class AquaDataset(Dataset):
    """PyTorch Dataset loading images from manifest DataFrames."""
    def __init__(self, df, label_to_idx=None, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        
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
        img_path = row['image_path']
        
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Failed to read image at: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        label_idx = self.label_to_idx[row['label']]
        return image, torch.tensor(label_idx, dtype=torch.long)
