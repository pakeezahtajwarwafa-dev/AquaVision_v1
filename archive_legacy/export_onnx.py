import torch
from pathlib import Path
from aquavision.models.vision_backbone import AquaVisionModel
import pandas as pd

def export_to_onnx(species="fish", backbone="efficientnet_b0", best_fold=3):
    # 1. Setup paths
    ckpt_path = Path(f"checkpoints/{backbone}_{species}_fold{best_fold}_best.pt")
    exports_dir = Path("exports")
    exports_dir.mkdir(exist_ok=True)
    onnx_path = exports_dir / f"{backbone}_{species}_optimized.onnx"

    if not ckpt_path.exists():
        print(f"[ERROR] Checkpoint not found at {ckpt_path}")
        return

    # 2. Get number of classes from the manifest
    csv_path = Path(f"datasets/processed/{species}_processed_manifest.csv")
    df = pd.read_csv(csv_path)
    num_classes = df[df['fold'] >= 0]['label'].nunique()

    print(f"\n--- Starting ONNX Export for {species.upper()} ---")
    
    # 3. Load PyTorch model
    device = torch.device("cpu") # Always export on CPU for maximum compatibility
    model = AquaVisionModel(num_classes=num_classes, backbone=backbone).to(device)
    
    # Using weights_only=True to silence the security warning
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()

    # 4. Create a dummy input tensor matching the image dimensions (Batch, Channels, Height, Width)
    dummy_input = torch.randn(1, 3, 224, 224).to(device)

    # 5. Export to ONNX
    print("Compiling PyTorch graph to ONNX...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,          # Opset 14 is highly stable for EfficientNet/ResNet
        do_constant_folding=True,  # Optimizes constant operations
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}} # Allows variable batch sizes later
    )
    
    print(f"[SUCCESS] Model optimized and saved to: {onnx_path}")

if __name__ == "__main__":
    export_to_onnx()
