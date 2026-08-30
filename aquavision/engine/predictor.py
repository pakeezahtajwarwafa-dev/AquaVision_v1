import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import numpy as np
import pandas as pd
import joblib
import cv2
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2

from aquavision.preprocessing import letterbox_resize
from aquavision.fusion import AquaVisionMultimodalModel

class AquaPredictor:
    def __init__(self, species: str = "fish", fold: int = 0):
        self.species = species
        self.fold = fold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        manifest_path = Path("datasets/processed") / f"{species}_processed_manifest.csv"
        df = pd.read_csv(manifest_path)
        self.classes = sorted(df["label"].unique())
        self.tab_cols = [c for c in ["temperature", "ph", "dissolved_oxygen", "ammonia", "temp_do_ratio", "ammonia_toxicity_index", "stress_score"] if c in df.columns]

        scaler_path = Path("checkpoints") / f"{species}_scaler_fold{fold}.pkl"
        self.scaler = joblib.load(scaler_path) if scaler_path.exists() else None

        checkpoint_path = Path("checkpoints") / f"{species}_best_model_fold{fold}.pt"
        self.model = AquaVisionMultimodalModel(num_classes=len(self.classes), num_tabular_features=len(self.tab_cols), backbone_name="resnet18").to(self.device)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device, weights_only=True))
        self.model.eval()

        # --- SPECIES GATE SETUP ---
        gate_path = Path("checkpoints/species_gate_best.pt")
        self.gate_classes = {0: "fish", 1: "shrimp"}
        self.species_gate = models.resnet18(weights=None)
        self.species_gate.fc = nn.Linear(self.species_gate.fc.in_features, 2)
        if gate_path.exists():
            self.species_gate.load_state_dict(torch.load(gate_path, map_location=self.device, weights_only=True))
        self.species_gate = self.species_gate.to(self.device)
        self.species_gate.eval()
        # --------------------------

        self.transform = A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

    def predict(self, image_input, water_quality_dict: dict = None):
        if isinstance(image_input, str):
            img = cv2.imread(image_input)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = image_input

        img_resized = letterbox_resize(img, target_size=224, species=self.species)
        img_tensor = self.transform(image=img_resized)["image"].unsqueeze(0).to(self.device)

        # --- 1. RUN SPECIES GATE ---
        with torch.no_grad():
            gate_logits = self.species_gate(img_tensor)
            gate_probs = F.softmax(gate_logits, dim=1).squeeze(0).cpu().numpy()
            gate_pred_idx = int(np.argmax(gate_probs))
            detected_species = self.gate_classes[gate_pred_idx]
            detected_conf = float(gate_probs[gate_pred_idx])

        # --- 2. VALIDATE SPECIES ---
        if detected_species != self.species.lower() and detected_conf > 0.65:
            return {
                "prediction": "Species Mismatch",
                "confidence": detected_conf,
                "all_probabilities": {},
                "is_mismatch": True,
                "message": f"Warning: You selected '{self.species}', but the model is {detected_conf*100:.1f}% confident this image is a {detected_species}."
            }

        # --- 3. RUN DISEASE PREDICTION ---
        with torch.no_grad():
            if self.tab_cols and water_quality_dict:
                wq = water_quality_dict.copy()
                wq["temp_do_ratio"] = wq.get("temperature", 25.0) / (wq.get("dissolved_oxygen", 6.0) + 1e-5)
                wq["ammonia_toxicity_index"] = wq.get("ammonia", 0.01) * (10 ** (wq.get("ph", 7.5) - 7.0)) * (wq.get("temperature", 25.0) / 25.0)
                wq["stress_score"] = (abs(wq.get("ph", 7.5) - 7.5) * 0.3) + (wq.get("ammonia", 0.01) * 10.0 * 0.4) + (max(0, 6.0 - wq.get("dissolved_oxygen", 6.0)) * 0.3)

                raw_df = pd.DataFrame([[wq[c] for c in self.tab_cols]], columns=self.tab_cols)
                scaled_vector = self.scaler.transform(raw_df)[0] if self.scaler else raw_df.values[0]
                tab_tensor = torch.tensor(scaled_vector, dtype=torch.float32).unsqueeze(0).to(self.device)
                logits = self.model(img_tensor, tab_tensor)
            else:
                logits = self.model(img_tensor)

        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        if probs.ndim == 0:
            probs = np.array([probs])
            
        top_idx = int(np.argmax(probs))
        return {
            "prediction": self.classes[top_idx],
            "confidence": float(probs[top_idx]),
            "all_probabilities": {cls: float(p) for cls, p in zip(self.classes, probs)},
            "is_mismatch": False
        }
