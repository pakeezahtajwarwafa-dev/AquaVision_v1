from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from aquavision.dataset import AquaDataset, get_transforms
from aquavision.models import build_model


class GradCAM:
    """Grad-CAM visual explainer for PyTorch CNN backbones."""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x, class_idx=None):
        self.model.eval()
        output = self.model(x)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        score = output[0, class_idx]
        score.backward()

        gradients = self.gradients.data.cpu().numpy()[0]
        activations = self.activations.data.cpu().numpy()[0]

        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = np.maximum(cam, 0)
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam, class_idx


def generate_evaluation_artifacts(model, loader, dataset_obj, device, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    target_names = [dataset_obj.idx_to_label[i] for i in range(len(dataset_obj.label_to_idx))]

    # Classification Report
    report = classification_report(all_targets, all_preds, target_names=target_names, output_dict=True, zero_division=0)
    df_report = pd.DataFrame(report).transpose()
    df_report.to_csv(out_dir / "classification_report.csv")

    # Confusion Matrix Plot
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=target_names, yticklabels=target_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix.png", dpi=300)
    plt.close()

    print(f"Evaluation artifacts saved to: {out_dir}")


def run(config, exp_name: str = "smoke_test", dataset_type: str = "fish"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_dir = config.paths.outputs / "experiments" / exp_name
    ckpt_path = exp_dir / "best_model.pth"

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    manifest_path = config.paths.datasets / "manifests" / f"{dataset_type}_manifest_deduped.csv"
    df = pd.read_csv(manifest_path)
    df_test = df[df["split"] == "test"].reset_index(drop=True)

    if df_test.empty:
        df_test = df[df["split"] == "val"].reset_index(drop=True)

    train_manifest = df[df["split"] == "train"].reset_index(drop=True)
    unique_labels = sorted(train_manifest["label"].unique())
    label_to_idx = {lbl: i for i, lbl in enumerate(unique_labels)}

    test_transform = get_transforms(img_size=config.training.img_size, is_train=False)
    test_ds = AquaDataset(df_test, label_to_idx=label_to_idx, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=config.training.batch_size, shuffle=False)

    model = build_model("resnet34", num_classes=len(label_to_idx), pretrained=False)
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model = model.to(device)

    generate_evaluation_artifacts(model, test_loader, test_ds, device, exp_dir)


if __name__ == "__main__":
    from config import load_config
    cfg = load_config()
    run(cfg, exp_name="smoke_test", dataset_type="fish")
