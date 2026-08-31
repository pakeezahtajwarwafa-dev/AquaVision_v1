import os
import shutil
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

def compute_dhash(image_path, hash_size=8):
    """Compute difference hash (dhash) using OpenCV."""
    try:
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        resized = cv2.resize(img, (hash_size + 1, hash_size))
        diff = resized[:, 1:] > resized[:, :-1]
        return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])
    except Exception:
        return None

def deduplicate_split_group(candidate_paths):
    """
    Sort duplicate candidates to prioritize high-res original images over 
    downscaled copies in Augmented_shrimp.
    """
    sorted_candidates = sorted(
        candidate_paths,
        key=lambda p: (
            # Priority 1 for Augmented_shrimp (deprioritized), 0 for original/unaugmented paths
            1 if "augmented_shrimp" in str(p).replace("\\", "/").lower() else 0,
            # Secondary sort by path string for deterministic tie-breaking
            str(p)
        )
    )
    # Index 0 is guaranteed to be the high-res original if one exists
    return sorted_candidates[0], sorted_candidates[1:]

def run_deduplication():
    raw_base = Path("datasets/raw")
    processed_base = Path("datasets/processed")

    if not raw_base.exists():
        print(f"Error: Directory '{raw_base}' does not exist.")
        return

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    fish_files = []
    shrimp_files = []

    # Recursively collect all raw images
    for full_path in raw_base.rglob("*"):
        if full_path.is_file() and full_path.suffix.lower() in image_extensions:
            path_str = str(full_path).replace("\\", "/").lower()
            if "shrimp" in path_str:
                shrimp_files.append(full_path)
            elif "fish" in path_str:
                fish_files.append(full_path)

    species_map = {
        "fish": fish_files,
        "shrimp": shrimp_files
    }

    for species, image_paths in species_map.items():
        print(f"\n--- Processing {species.upper()} Dataset ---")
        print(f"Found {len(image_paths)} total images for {species}.")

        if not image_paths:
            print(f"No images matched species '{species}' in {raw_base}.")
            continue

        # 1. Group images by perceptual hash (dhash)
        hash_map = defaultdict(list)
        for img_path in image_paths:
            img_hash = compute_dhash(img_path)
            if img_hash is not None:
                hash_map[img_hash].append(img_path)

        # 2. Select keepers and identify duplicates using priority rules
        kept_images = []
        total_discarded = 0

        for img_hash, candidates in hash_map.items():
            if len(candidates) == 1:
                kept_images.append(candidates[0])
            else:
                kept, discarded = deduplicate_split_group(candidates)
                kept_images.append(kept)
                total_discarded += len(discarded)

        print(f"Deduplication complete for {species}:")
        print(f"  - Kept (High-Res Originals): {len(kept_images)}")
        print(f"  - Discarded (Duplicates/Downscaled): {total_discarded}")

        # 3. Write clean dataset to processed directory
        out_dir = processed_base / species
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        manifest_rows = []
        for idx, src_path in enumerate(kept_images):
            # Extract parent folder name for disease label (e.g. White spot, Healthy, EUS)
            parent_name = src_path.parent.name
            if parent_name.lower() in ["shrimpimages", "augmented_shrimp", "fishimages", "fish", "shrimp", "raw"]:
                class_label = "healthy"
            else:
                class_label = parent_name

            dest_filename = f"{species}_{idx:05d}{src_path.suffix}"
            dest_path = out_dir / dest_filename

            shutil.copy2(src_path, dest_path)
            manifest_rows.append({
                "image_path": dest_filename,
                "label": class_label,
                "original_source": str(src_path)
            })

        # 4. Save clean manifest
        manifest_df = pd.DataFrame(manifest_rows)
        manifest_path = processed_base / f"{species}_processed_manifest.csv"
        manifest_df.to_csv(manifest_path, index=False)
        print(f"Saved clean manifest to: {manifest_path}")

if __name__ == "__main__":
    run_deduplication()
