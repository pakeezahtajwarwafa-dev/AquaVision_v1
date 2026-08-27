import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2

from aquavision.preprocessing import letterbox_resize

def get_transforms(is_train: bool = True):
    """Returns Albumentations transformation pipeline tailored for underwater imaging."""
    if is_train:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=30, p=0.7, border_mode=cv2.BORDER_CONSTANT),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.6),
            A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=20, p=0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

class AquaDataset(Dataset):
    """PyTorch Dataset loading letterboxed images & optional scaled water metrics."""
    def __init__(self, df: pd.DataFrame, img_size: int = 224, is_train: bool = True, species: str = "fish", tab_cols: list = None):
        self.df = df.reset_index(drop=True)
        self.img_size = img_size
        self.species = species
        self.tab_cols = tab_cols or []
        self.transforms = get_transforms(is_train=is_train)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["image_path"]

        img = cv2.imread(img_path)
        if img is None:
            # Fallback zero array if corrupt image
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_resized = letterbox_resize(img, target_size=self.img_size, species=self.species)
        augmented = self.transforms(image=img_resized)
        img_tensor = augmented["image"]

        label = int(row["label_idx"]) if "label_idx" in row else 0

        if self.tab_cols:
            tab_vector = row[self.tab_cols].values.astype(np.float32)
            tab_tensor = torch.tensor(tab_vector, dtype=torch.float32)
            return img_tensor, tab_tensor, torch.tensor(label, dtype=torch.long)

        return img_tensor, torch.tensor(label, dtype=torch.long)
