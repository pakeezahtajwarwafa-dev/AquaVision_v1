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
        "Fungal diseases Saprolegniasis",  # <-- Added missing fungal class
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

        # Recover held-out test split
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
            "image_name": filename,
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

    # Ensure there is training data to split
    if train_mask.sum() > 0:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        train_df = df[train_mask].reset_index(drop=True)

        for fold_idx, (_, val_idx) in enumerate(skf.split(train_df, train_df["label"])):
            train_df.loc[val_idx, "fold"] = fold_idx

        df.loc[train_mask, "fold"] = train_df["fold"].values

    # NOTE: Tabular water quality merge block completely removed

    out_path = processed_dir / f"{species}_processed_manifest.csv"
    df.to_csv(out_path, index=False)
    print(f"[{species.upper()}] Manifest saved to {out_path} with {df['fold'].nunique() - 1} train folds + test split.")


if __name__ == "__main__":
    for sp in ["fish", "shrimp"]:
        build_robust_manifest(species=sp)