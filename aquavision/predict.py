import torch
import torch.nn.functional as F
import cv2
import pandas as pd
import numpy as np
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2

from aquavision.preprocessing import letterbox_resize
from aquavision.fusion import AquaVisionMultimodalModel

class AquaPredictor:
    """Inference engine for multimodal aquatic disease diagnosis."""
    def __init__(self, species: str = "fish", fold: int = 0):
        self.species = species
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load processed manifest to extract class names and tabular columns
        manifest_path = Path("datasets/processed") / f"{species}_processed_manifest.csv"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest missing: {manifest_path}")

        df = pd.read_csv(manifest_path)
        self.classes = sorted(df["label"].unique())
        self.tab_cols = [c for c in ["temperature", "ph", "dissolved_oxygen", "ammonia", "temp_do_ratio", "ammonia_toxicity_index", "stress_score"] if c in df.columns]

        # Model initialization
        checkpoint_path = Path("checkpoints") / f"{species}_best_model_fold{fold}.pt"
        self.model = AquaVisionMultimodalModel(
            num_classes=len(self.classes),
            num_tabular_features=len(self.tab_cols),
            backbone_name="resnet18"
        ).to(self.device)

        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device, weights_only=True))
        self.model.eval()

        self.transform = A.Compose([
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

    def predict(self, image_path: str, water_quality_dict: dict = None):
        """Predicts disease class and confidence score for a given image and water sample."""
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = letterbox_resize(img, target_size=224, species=self.species)
        img_tensor = self.transform(image=img)["image"].unsqueeze(0).to(self.device)

        if self.tab_cols and water_quality_dict:
            # Engineer features if base water quality parameters exist
            temp = water_quality_dict.get("temperature", 25.0)
            do = water_quality_dict.get("dissolved_oxygen", 6.0)
            ph = water_quality_dict.get("ph", 7.5)
            ammonia = water_quality_dict.get("ammonia", 0.05)

            water_quality_dict["temp_do_ratio"] = temp / (do + 1e-5)
            water_quality_dict["ammonia_toxicity_index"] = ammonia * (10 ** (ph - 7.0)) * (temp / 25.0)
            water_quality_dict["stress_score"] = (abs(ph - 7.5) * 0.3) + (ammonia * 10.0 * 0.4) + (max(0, 6.0 - do) * 0.3)

            tab_vector = [float(water_quality_dict.get(c, 0.0)) for c in self.tab_cols]
            tab_tensor = torch.tensor([tab_vector], dtype=torch.float32).to(self.device)
            
            with torch.no_grad():
                logits = self.model(img_tensor, tab_tensor)
        else:
            with torch.no_grad():
                logits = self.model(img_tensor)

        probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()
        top_idx = int(np.argmax(probs))

        return {
            "predicted_class": self.classes[top_idx],
            "confidence": float(probs[top_idx]),
            "class_probabilities": {self.classes[i]: float(probs[i]) for i in range(len(self.classes))}
        }

if __name__ == "__main__":
    # Quick test on first available image in fish manifest
    df_fish = pd.read_csv("datasets/processed/fish_processed_manifest.csv")
    sample_img = df_fish.iloc[0]["image_path"]
    
    predictor = AquaPredictor(species="fish", fold=0)
    sample_wq = {"temperature": 28.5, "ph": 8.2, "dissolved_oxygen": 4.1, "ammonia": 0.45}
    result = predictor.predict(sample_img, sample_wq)

    print(f"\n--- SINGLE INFERENCE TEST ---")
    print(f"Sample Image   : {sample_img}")
    print(f"Diagnosis      : {result['predicted_class']}")
    print(f"Confidence     : {result['confidence'] * 100:.2f}%")
