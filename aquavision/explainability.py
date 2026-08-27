import torch
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2

from aquavision.preprocessing import letterbox_resize
from aquavision.fusion import AquaVisionMultimodalModel

class AquaGradCAM:
    """Generates Class Activation Maps (Grad-CAM) for disease diagnostics."""
    def __init__(self, species: str = "fish", fold: int = 0):
        self.species = species
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        manifest_path = Path("datasets/processed") / f"{species}_processed_manifest.csv"
        df = pd.read_csv(manifest_path)
        self.classes = sorted(df["label"].unique())
        self.tab_cols = [c for c in ["temperature", "ph", "dissolved_oxygen", "ammonia", "temp_do_ratio", "ammonia_toxicity_index", "stress_score"] if c in df.columns]

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

        # Hook targets final ResNet layer directly via self.model.backbone
        self.gradients = None
        self.activations = None
        target_layer = self.model.backbone.layer4[-1]
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, image_path: str, water_quality_dict: dict = None, output_path: str = "datasets/gradcam_result.jpg"):
        orig_img = cv2.imread(image_path)
        if orig_img is None:
            raise FileNotFoundError(f"Image not found: {image_path}")

        rgb_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        resized_img = letterbox_resize(rgb_img, target_size=224, species=self.species)
        img_tensor = self.transform(image=resized_img)["image"].unsqueeze(0).to(self.device)

        if self.tab_cols and water_quality_dict:
            temp = water_quality_dict.get("temperature", 25.0)
            do = water_quality_dict.get("dissolved_oxygen", 6.0)
            ph = water_quality_dict.get("ph", 7.5)
            ammonia = water_quality_dict.get("ammonia", 0.05)

            water_quality_dict["temp_do_ratio"] = temp / (do + 1e-5)
            water_quality_dict["ammonia_toxicity_index"] = ammonia * (10 ** (ph - 7.0)) * (temp / 25.0)
            water_quality_dict["stress_score"] = (abs(ph - 7.5) * 0.3) + (ammonia * 10.0 * 0.4) + (max(0, 6.0 - do) * 0.3)

            tab_vector = [float(water_quality_dict.get(c, 0.0)) for c in self.tab_cols]
            tab_tensor = torch.tensor([tab_vector], dtype=torch.float32).to(self.device)
            logits = self.model(img_tensor, tab_tensor)
        else:
            logits = self.model(img_tensor)

        pred_idx = torch.argmax(logits, dim=1).item()
        pred_label = self.classes[pred_idx]

        self.model.zero_grad()
        target_score = logits[0, pred_idx]
        target_score.backward()

        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations[0]

        for i in range(activations.size(0)):
            activations[i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(activations, dim=0).cpu().detach().numpy()
        heatmap = np.maximum(heatmap, 0)
        if np.max(heatmap) != 0:
            heatmap /= np.max(heatmap)

        heatmap_resized = cv2.resize(heatmap, (224, 224))
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        
        overlay = cv2.addWeighted(cv2.cvtColor(resized_img, cv2.COLOR_RGB2BGR), 0.6, heatmap_colored, 0.4, 0)
        
        cv2.imwrite(output_path, overlay)
        print(f"  > Generated Grad-CAM heatmap for class '{pred_label}' -> Saved to: {output_path}")
        return pred_label, output_path

if __name__ == "__main__":
    df_fish = pd.read_csv("datasets/processed/fish_processed_manifest.csv")
    sample_img = df_fish.iloc[0]["image_path"]

    cam_engine = AquaGradCAM(species="fish", fold=0)
    sample_wq = {"temperature": 28.5, "ph": 8.2, "dissolved_oxygen": 4.1, "ammonia": 0.45}
    cam_engine.generate_heatmap(sample_img, sample_wq)
