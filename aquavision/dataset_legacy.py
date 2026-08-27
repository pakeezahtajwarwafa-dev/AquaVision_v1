import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from aquavision.preprocessing import letterbox_resize

class AquaDataset(Dataset):
    """PyTorch Dataset with auto label-encoding, letterbox padding, and multimodal feature support."""
    def __init__(
        self, 
        df: pd.DataFrame, 
        img_size: int = 224, 
        is_train: bool = True, 
        species: str = "fish",
        tab_cols: list = None
    ):
        self.df = df.reset_index(drop=True).copy()
        self.img_size = img_size
        self.is_train = is_train
        self.species = species
        self.tab_cols = tab_cols if tab_cols is not None else []
        
        # Auto-encode string labels to integer indices if label_idx is absent
        if "label_idx" not in self.df.columns and "label" in self.df.columns:
            if self.df["label"].dtype == object or isinstance(self.df["label"].iloc[0], str):
                unique_labels = sorted(self.df["label"].unique())
                label2idx = {lbl: idx for idx, lbl in enumerate(unique_labels)}
                self.df["label_idx"] = self.df["label"].map(label2idx)
            else:
                self.df["label_idx"] = self.df["label"]

        # Extract and normalize tabular features if provided
        if self.tab_cols:
            self.tab_data = self.df[self.tab_cols].fillna(0.0).values.astype(np.float32)
        else:
            self.tab_data = None

        # Base normalization for validation/test & pre-augmented images
        self.base_transform = A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
        
        # Augmentation pipeline for un-augmented training images
        self.geo_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, p=0.3),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = str(row["image_path"])
        
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image read error: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Apply species-aware letterbox padding + square resize
        img = letterbox_resize(img, target_size=self.img_size, species=self.species)
        
        # Skip geometric transforms on pre-augmented photos
        is_aug = row.get("is_augmented", False)
        if self.is_train and not is_aug:
            transformed = self.geo_transform(image=img)
        else:
            transformed = self.base_transform(image=img)
            
        image_tensor = transformed["image"]
        label = int(row.get("label_idx", 0))
        
        if self.tab_data is not None:
            tab_tensor = torch.tensor(self.tab_data[idx], dtype=torch.float32)
            return image_tensor, tab_tensor, torch.tensor(label, dtype=torch.long)
        
        return image_tensor, torch.tensor(label, dtype=torch.long)
