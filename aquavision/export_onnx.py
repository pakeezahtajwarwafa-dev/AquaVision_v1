import torch
from pathlib import Path
import pandas as pd

from aquavision.fusion import AquaVisionMultimodalModel

def export_species_onnx(species: str = "fish", fold: int = 0):
    device = torch.device("cpu")
    checkpoint_path = Path("checkpoints") / f"{species}_best_model_fold{fold}.pt"
    manifest_path = Path("datasets/processed") / f"{species}_processed_manifest.csv"

    if not checkpoint_path.exists():
        print(f"⚠️ Checkpoint missing for {species}: {checkpoint_path}")
        return

    # Load state dict directly to inspect dimensions
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)

    # 1. Dynamically inspect tabular input feature count from weights
    if "tabular_mlp.net.0.weight" in state_dict:
        num_tabular_features = state_dict["tabular_mlp.net.0.weight"].shape[1]
    else:
        num_tabular_features = 0

    # 2. Dynamically inspect class count from output layer weights
    classifier_weight_keys = [k for k in state_dict.keys() if "classifier" in k and "weight" in k]
    if classifier_weight_keys:
        num_classes = state_dict[classifier_weight_keys[-1]].shape[0]
    else:
        num_classes = 8 # Fallback standard

    print(f"[{species.upper()}] Detected Checkpoint Specs -> Num Classes: {num_classes}, Tabular Features: {num_tabular_features}")

    # Build exact matching model instance
    model = AquaVisionMultimodalModel(
        num_classes=num_classes,
        num_tabular_features=num_tabular_features,
        backbone_name="resnet18"
    ).to(device)

    model.load_state_dict(state_dict)
    model.eval()

    # Build dummy input tensors matching checkpoint configuration
    dummy_image = torch.randn(1, 3, 224, 224, device=device)
    export_dir = Path("exports")
    export_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = export_dir / f"{species}_model_fold{fold}.onnx"

    if num_tabular_features > 0:
        dummy_tabular = torch.randn(1, num_tabular_features, device=device)
        dummy_inputs = (dummy_image, dummy_tabular)
        input_names = ["image_input", "tabular_input"]
        dynamic_axes = {
            "image_input": {0: "batch_size"},
            "tabular_input": {0: "batch_size"},
            "logits": {0: "batch_size"}
        }
    else:
        dummy_inputs = (dummy_image,)
        input_names = ["image_input"]
        dynamic_axes = {
            "image_input": {0: "batch_size"},
            "logits": {0: "batch_size"}
        }

    torch.onnx.export(
        model,
        dummy_inputs,
        str(onnx_path),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=input_names,
        output_names=["logits"],
        dynamic_axes=dynamic_axes
    )

    print(f"✅ [{species.upper()}] Successfully exported ONNX model to {onnx_path}\n")

if __name__ == "__main__":
    for sp in ["fish", "shrimp"]:
        export_species_onnx(species=sp, fold=0)
