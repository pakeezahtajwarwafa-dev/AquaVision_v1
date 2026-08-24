import torch
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from sklearn.preprocessing import StandardScaler

from aquavision.dataset import get_transforms
from aquavision.fusion import FusionModel


class AquaPredictor:
    """Unified multimodal inference pipeline for real-time disease diagnosis."""
    def __init__(self, config, dataset_type="fish"):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load Manifest to reconstruct class mappings
        manifest_path = config.paths.datasets / "manifests" / f"{dataset_type}_manifest_deduped.csv"
        df_manifest = pd.read_csv(manifest_path)
        train_manifest = df_manifest[df_manifest["split"] == "train"].reset_index(drop=True)
        unique_labels = sorted(train_manifest["label"].unique())
        
        self.label_to_idx = {lbl: i for i, lbl in enumerate(unique_labels)}
        self.idx_to_label = {v: k for k, v in self.label_to_idx.items()}
        num_classes = len(unique_labels)

        # Fit Tabular Scaler on historical water quality data
        wq_path = config.paths.datasets / "water_quality" / f"{dataset_type}_water_quality.csv"
        df_wq = pd.read_csv(wq_path)
        tab_cols = ["ph", "temperature", "dissolved_oxygen", "salinity", "ammonia", "turbidity"]
        
        self.scaler = StandardScaler()
        self.scaler.fit(df_wq[tab_cols].values.astype(np.float32))

        # Load Fusion Model Checkpoint
        ckpt_path = config.paths.outputs / "experiments" / f"{dataset_type}_multimodal_fusion" / "best_fusion_model.pth"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

        self.model = FusionModel("resnet34", num_classes=num_classes, pretrained=False).to(self.device)
        self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device, weights_only=True))
        self.model.eval()

        self.transform = get_transforms(img_size=config.training.img_size, is_train=False)

    def predict(self, image_input, ph: float, temp: float, do: float, salinity: float, ammonia: float, turbidity: float):
        # Image Preprocessing
        if isinstance(image_input, str) or isinstance(image_input, Path):
            img = cv2.imread(str(image_input))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = image_input  # RGB NumPy array from Gradio

        transformed = self.transform(image=img)["image"].unsqueeze(0).to(self.device)

        # Tabular Preprocessing
        tab_raw = np.array([[ph, temp, do, salinity, ammonia, turbidity]], dtype=np.float32)
        tab_scaled = torch.tensor(self.scaler.transform(tab_raw), dtype=torch.float32).to(self.device)

        # Forward Pass
        with torch.no_grad():
            outputs = self.model(transformed, tab_scaled)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

        predictions = {self.idx_to_label[i]: float(probs[i]) for i in range(len(probs))}
        top_pred = self.idx_to_label[np.argmax(probs)]
        confidence = float(np.max(probs))

        return predictions, top_pred, confidence


if __name__ == "__main__":
    from config import load_config
    cfg = load_config()
    predictor = AquaPredictor(cfg, dataset_type="fish")
    print("Inference Engine loaded successfully!")
