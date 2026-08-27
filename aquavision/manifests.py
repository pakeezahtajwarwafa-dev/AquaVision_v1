import os
from pathlib import Path
import pandas as pd
from config import load_config

def find_species_dir(base_paths, species_name: str) -> Path:
    """Finds existing species directory across common dataset folder structures."""
    for base in base_paths:
        if base is None:
            continue
        base = Path(base)
        # Direct check: root/species
        candidate = base / species_name
        if candidate.exists() and candidate.is_dir():
            return candidate
        # Nested check: root/raw/species
        candidate_raw = base / "raw" / species_name
        if candidate_raw.exists() and candidate_raw.is_dir():
            return candidate_raw
    return None

def build_manifest_for_species(species_dir: Path, species_name: str) -> pd.DataFrame:
    records = []
    if species_dir is None or not species_dir.exists():
        print(f"Warning: Directory for '{species_name}' could not be located.")
        return pd.DataFrame()

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for root, _, files in os.walk(species_dir):
        for file in files:
            ext = Path(file).suffix.lower()
            if ext in valid_exts:
                full_path = Path(root) / file
                class_label = full_path.parent.name
                
                parts = [p.lower() for p in full_path.parts]
                if "val" in parts or "validation" in parts:
                    split = "val"
                elif "test" in parts:
                    split = "test"
                else:
                    split = "train"

                is_aug = file.lower().startswith("aug_") or "_aug_" in file.lower()

                records.append({
                    "image_path": str(full_path.resolve()),
                    "label": class_label,
                    "species": species_name,
                    "split": split,
                    "is_augmented": is_aug
                })

    df = pd.DataFrame(records)
    print(f"[{species_name.capitalize()}] Found {len(df)} images across {df['label'].nunique() if not df.empty else 0} classes.")
    return df

def run(config):
    manifest_dir = Path("datasets/manifests")
    if hasattr(config.paths, "datasets"):
        manifest_dir = config.paths.datasets / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    
    # Candidate root folders to search
    candidate_roots = [
        Path("data"),
        Path("datasets"),
        getattr(config.paths, "datasets", None),
        Path("."),
    ]

    for species in ["fish", "shrimp"]:
        species_dir = find_species_dir(candidate_roots, species)
        if species_dir:
            print(f"Located '{species}' folder at: {species_dir.resolve()}")
        df_manifest = build_manifest_for_species(species_dir, species)
        
        if not df_manifest.empty:
            out_file = manifest_dir / f"{species}_manifest.csv"
            df_manifest.to_csv(out_file, index=False)
            print(f"Saved regenerated manifest: {out_file}\n")

if __name__ == "__main__":
    cfg = load_config()
    run(cfg)
