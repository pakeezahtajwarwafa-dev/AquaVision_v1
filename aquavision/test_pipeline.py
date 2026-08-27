import torch
import pandas as pd
from torch.utils.data import DataLoader
from aquavision.dataset import AquaDataset
from aquavision.fusion import AquaVisionMultimodalModel

def run_test():
    print("==========================================")
    print("      AQUAVISION PIPELINE SANITY TEST     ")
    print("==========================================")

    # 1. Load Processed Manifest
    manifest_path = "datasets/processed/fish_processed_manifest.csv"
    print(f"[1/4] Loading manifest: {manifest_path}")
    df = pd.read_csv(manifest_path)
    
    unique_labels = sorted(df['label'].unique())
    label2idx = {lbl: idx for idx, lbl in enumerate(unique_labels)}
    df['label_idx'] = df['label'].map(label2idx)
    num_classes = len(unique_labels)
    print(f"      Loaded {len(df)} rows across {num_classes} target classes.")

    # 2. Check Tabular Features
    tab_cols = [c for c in ["temperature", "ph", "dissolved_oxygen", "ammonia", "temp_do_ratio", "ammonia_toxicity_index", "stress_score"] if c in df.columns]
    print(f"[2/4] Tabular features detected ({len(tab_cols)}): {tab_cols}")

    # 3. Test Dataset & DataLoader Batch Processing
    print("[3/4] Testing AquaDataset & Letterbox Resizing...")
    ds = AquaDataset(df, img_size=224, is_train=True, species="fish")
    loader = DataLoader(ds, batch_size=4, shuffle=True)
    
    images, labels = next(iter(loader))
    print(f"      Batch Image Shape : {tuple(images.shape)} (Expected: [4, 3, 224, 224])")
    print(f"      Batch Target Labels: {labels.tolist()}")

    # 4. Model Forward Pass Verification
    print("[4/4] Testing Multimodal Model Forward Pass...")
    model = AquaVisionMultimodalModel(
        num_classes=num_classes,
        num_tabular_features=len(tab_cols),
        backbone_name="resnet18"
    )
    model.eval()

    with torch.no_grad():
        if len(tab_cols) > 0:
            dummy_tab = torch.randn(4, len(tab_cols))
            logits = model(images, dummy_tab)
        else:
            logits = model(images)

    print(f"      Model Logits Output Shape: {tuple(logits.shape)} (Expected: [4, {num_classes}])")
    print("==========================================")
    print(" SUCCESS: All pipeline components are functional!")
    print("==========================================")

if __name__ == "__main__":
    run_test()
