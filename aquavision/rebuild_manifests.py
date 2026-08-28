import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold

def build_robust_manifest(species: str = "fish"):
    raw_dir = Path("datasets") / species
    processed_dir = Path("datasets/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    records = []
    classes = set()

    # Known class keywords for parsing test_split filenames
    known_classes = [
        "Bacterial diseases - Aeromoniasis",
        "Bacterial gill disease",
        "Bacterial Red disease",
        "EUS",
        "Healthy Fish",
        "Parasitic diseases",
        "Viral diseases White tail disease",
        "BG", "Healthy", "WSSV", "WSSV_BG"
    ]

    for img_path in raw_dir.rglob("*.*"):
        if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp"]:
            continue

        filename = img_path.name
        rel_parts = img_path.parts

        # Bug 4.2 Fix: Recover held-out test split
        is_test = "test_split" in rel_parts
        assigned_label = None

        if is_test:
            for cls in sorted(known_classes, key=len, reverse=True):
                if filename.startswith(cls):
                    assigned_label = cls
                    break
            if not assigned_label:
                continue
        else:
            assigned_label = img_path.parent.name

        # Exclude fake/temp split labels
        if assigned_label in ["test_split", "train_split", species]:
            continue

        classes.add(assigned_label)
        records.append({
            "image_name": filename,  # Bug 4.1 Fix: Canonical filename key
            "image_path": str(img_path.resolve()),
            "label": assigned_label,
            "is_test": is_test,
            "is_augmented": "aug" in filename.lower()
        })

    df = pd.DataFrame(records)
    print(f"[{species.upper()}] Parsed {len(df)} images ({df['is_test'].sum()} held-out test samples)")

    # Assign K-Fold to training data, -1 to test set
    df["fold"] = -1
    train_mask = ~df["is_test"]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    train_df = df[train_mask].reset_index(drop=True)
    for fold_idx, (_, val_idx) in enumerate(skf.split(train_df, train_df["label"])):
        train_df.loc[val_idx, "fold"] = fold_idx

    df.loc[train_mask, "fold"] = train_df["fold"].values

    # Bug 4.1 Fix: Join water quality metrics on filename key
    wq_path = raw_dir / f"{species}_water_quality.csv"
    if wq_path.exists():
        wq_df = pd.read_csv(wq_path)
        if "image_name" not in wq_df.columns and "image_path" in wq_df.columns:
            wq_df["image_name"] = wq_df["image_path"].apply(lambda x: Path(x).name)
        
        # Select tabular columns
        tab_cols = [c for c in wq_df.columns if c not in ["image_path", "image_name", "label"]]
        df = df.merge(wq_df[["image_name"] + tab_cols], on="image_name", how="left")

    out_path = processed_dir / f"{species}_processed_manifest.csv"
    df.to_csv(out_path, index=False)
    print(f"[{species.upper()}] Manifest saved to {out_path} with {df['fold'].nunique()-1} train folds + test split.")

if __name__ == "__main__":
    for sp in ["fish", "shrimp"]:
        build_robust_manifest(species=sp)
