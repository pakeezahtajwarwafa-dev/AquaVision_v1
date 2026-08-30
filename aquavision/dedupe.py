import pandas as pd
from pathlib import Path
from PIL import Image
from config import load_config

def compute_dhash(image_path: str, hash_size: int = 8) -> int:
    """Calculates a 64-bit difference hash using PIL to detect near-duplicate images."""
    try:
        with Image.open(image_path) as img:
            img = img.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
            pixels = list(img.getdata())
            difference = []
            for row in range(hash_size):
                for col in range(hash_size):
                    pixel_left = pixels[row * (hash_size + 1) + col]
                    pixel_right = pixels[row * (hash_size + 1) + col + 1]
                    difference.append(pixel_left > pixel_right)
            decimal_val = 0
            for bit in difference:
                decimal_val = (decimal_val << 1) | bit
            return decimal_val
    except Exception:
        return None

def hamming_distance(h1: int, h2: int) -> int:
    """Calculates Hamming distance between two integer hashes."""
    return bin(h1 ^ h2).count("1")

def deduplicate_split_group(df_split: pd.DataFrame, max_hamming_dist: int = 4) -> tuple[pd.DataFrame, int]:
    """Deduplicates rows within a single split dataframe."""
    if df_split.empty:
        return df_split, 0

    df = df_split.copy().reset_index(drop=True)
    df["dhash"] = df["image_path"].apply(compute_dhash)

    # Deprioritize Augmented_shrimp: when a duplicate/near-duplicate pair is
    # found, the loop below keeps whichever row it sees first. Sorting these
    # paths to the end ensures the higher-quality original (e.g. ShrimpImages)
    # is kept and the lower-resolution Augmented_shrimp copy is dropped.
    df["_low_priority"] = df["image_path"].str.contains("Augmented_shrimp", regex=False)
    df = df.sort_values("_low_priority", kind="stable").reset_index(drop=True)

    valid_df = df.dropna(subset=["dhash"]).copy()
    hashes = valid_df["dhash"].tolist()
    indices = valid_df.index.tolist()

    keep_indices = []
    suppressed = set()
    dropped_count = 0

    for i in range(len(hashes)):
        if indices[i] in suppressed:
            continue
        keep_indices.append(indices[i])
        for j in range(i + 1, len(hashes)):
            if indices[j] in suppressed:
                continue
            if hamming_distance(hashes[i], hashes[j]) <= max_hamming_dist:
                suppressed.add(indices[j])
                dropped_count += 1

    clean_df = valid_df.loc[keep_indices].drop(columns=["dhash", "_low_priority"])
    return clean_df, dropped_count

def run(config):
    manifest_dir = Path("datasets/manifests")
    if hasattr(config.paths, "datasets"):
        manifest_dir = config.paths.datasets / "manifests"

    for species in ["fish", "shrimp"]:
        manifest_file = manifest_dir / f"{species}_manifest.csv"
        if not manifest_file.exists():
            continue

        df = pd.read_csv(manifest_file)
        initial_count = len(df)

        deduped_splits = []
        total_dropped = 0

        # Perform split-isolated deduplication
        for split_name, group in df.groupby("split"):
            clean_split, dropped = deduplicate_split_group(group)
            deduped_splits.append(clean_split)
            total_dropped += dropped

        df_deduped = pd.concat(deduped_splits, ignore_index=True)
        out_file = manifest_dir / f"{species}_manifest_deduped.csv"
        df_deduped.to_csv(out_file, index=False)
        print(f"[{species.capitalize()}] Initial: {initial_count} | Removed: {total_dropped} duplicates | Kept: {len(df_deduped)}")
        print(f"Saved split-aware deduped manifest: {out_file}\n")

if __name__ == "__main__":
    cfg = load_config()
    run(cfg)
