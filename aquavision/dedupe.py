from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
import imagehash


def deduplicate_dataframe(df: pd.DataFrame, threshold: int = 6):
    """
    Computes perceptual hashes (pHash) for all images in the manifest and removes
    near-duplicates based on matrix-vectorized Hamming distance comparisons.
    """
    hashes = []
    valid_indices = []

    for idx, row in df.iterrows():
        try:
            with Image.open(row["image_path"]) as img:
                h = imagehash.phash(img)
                hashes.append(h.hash.flatten())
                valid_indices.append(idx)
        except Exception:
            continue

    if not hashes:
        return df.copy(), 0

    df_clean = df.loc[valid_indices].copy().reset_index(drop=True)
    matrix = np.array(hashes, dtype=np.int32)

    # Vectorized Hamming Distance: ||a - b||1 = sum(a) + sum(b) - 2*(a . b)
    sums = matrix.sum(axis=1)
    dot = matrix @ matrix.T
    dist_matrix = sums[:, None] + sums[None, :] - 2 * dot

    n = len(df_clean)
    to_drop = set()

    for i in range(n):
        if i in to_drop:
            continue
        matches = np.where(dist_matrix[i, i + 1 :] <= threshold)[0] + (i + 1)
        for j in matches:
            to_drop.add(j)

    df_deduped = df_clean.drop(index=list(to_drop)).reset_index(drop=True)
    return df_deduped, len(to_drop)


def run(config):
    """Entry point: processes fish & shrimp manifests and outputs deduplicated manifests."""
    manifest_dir = config.paths.datasets / "manifests"
    threshold = config.manifests.hamming_threshold

    fish_in = manifest_dir / "fish_manifest.csv"
    shrimp_in = manifest_dir / "shrimp_manifest.csv"

    fish_out = manifest_dir / "fish_manifest_deduped.csv"
    shrimp_out = manifest_dir / "shrimp_manifest_deduped.csv"

    fish_removed, shrimp_removed = 0, 0
    df_fish_deduped, df_shrimp_deduped = pd.DataFrame(), pd.DataFrame()

    if fish_in.exists():
        df_fish = pd.read_csv(fish_in)
        df_fish_deduped, fish_removed = deduplicate_dataframe(
            df_fish, threshold=threshold
        )
        df_fish_deduped.to_csv(fish_out, index=False)

    if shrimp_in.exists():
        df_shrimp = pd.read_csv(shrimp_in)
        df_shrimp_deduped, shrimp_removed = deduplicate_dataframe(
            df_shrimp, threshold=threshold
        )
        df_shrimp_deduped.to_csv(shrimp_out, index=False)

    return {
        "fish_deduped_path": fish_out,
        "shrimp_deduped_path": shrimp_out,
        "fish_original": len(pd.read_csv(fish_in)) if fish_in.exists() else 0,
        "fish_clean": len(df_fish_deduped),
        "fish_removed": fish_removed,
        "shrimp_original": len(pd.read_csv(shrimp_in)) if shrimp_in.exists() else 0,
        "shrimp_clean": len(df_shrimp_deduped),
        "shrimp_removed": shrimp_removed,
    }


if __name__ == "__main__":
    from config import load_config

    cfg = load_config()
    res = run(cfg)
    print(
        f"Deduplication Complete:\n"
        f" - Fish: {res['fish_clean']} saved, {res['fish_removed']} near-duplicates removed.\n"
        f" - Shrimp: {res['shrimp_clean']} saved, {res['shrimp_removed']} near-duplicates removed."
    )
