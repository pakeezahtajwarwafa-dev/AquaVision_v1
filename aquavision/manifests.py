import hashlib
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


def hash_file(path: Path) -> str:
    """Computes MD5 hash of an image file to catch exact byte duplicates."""
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def build_fish_manifest(config) -> pd.DataFrame:
    base_dir = config.paths.datasets / "fish" / "New Dataset"
    train_dir = base_dir / "train_split"
    test_dir = base_dir / "test_split"
    img_exts = tuple(config.manifests.img_exts)
    seed = config.manifests.random_seed

    rows, seen_hashes = [], set()
    dropped_dupes = 0

    if train_dir.exists():
        for cls_dir in sorted(train_dir.iterdir()):
            if not cls_dir.is_dir():
                continue
            for fpath in cls_dir.iterdir():
                if fpath.suffix.lower() not in img_exts:
                    continue
                h = hash_file(fpath)
                if h in seen_hashes:
                    dropped_dupes += 1
                    continue
                seen_hashes.add(h)
                rows.append(
                    {
                        "image_path": str(fpath.resolve()),
                        "label": cls_dir.name,
                        "dataset": "fish",
                        "split": "train_full",
                    }
                )

    class_names = (
        [d.name for d in sorted(train_dir.iterdir()) if d.is_dir()]
        if train_dir.exists()
        else []
    )
    test_rows = []

    if test_dir.exists():
        for fpath in test_dir.iterdir():
            if fpath.suffix.lower() not in img_exts:
                continue
            matched = next(
                (cls for cls in class_names if fpath.name.startswith(cls)), None
            )
            if matched:
                test_rows.append(
                    {
                        "image_path": str(fpath.resolve()),
                        "label": matched,
                        "dataset": "fish",
                        "split": "test",
                    }
                )

    df_train_full = pd.DataFrame(rows)
    if not df_train_full.empty:
        train_idx, val_idx = train_test_split(
            df_train_full.index,
            test_size=0.15,
            stratify=df_train_full["label"],
            random_state=seed,
        )
        df_train_full.loc[train_idx, "split"] = "train"
        df_train_full.loc[val_idx, "split"] = "val"

    df_test = pd.DataFrame(test_rows)
    df_fish = pd.concat([df_train_full, df_test], ignore_index=True)
    return df_fish


def build_shrimp_manifest(config) -> pd.DataFrame:
    shrimp_dir = config.paths.datasets / "shrimp" / "ShrimpImages" / "ShrimpImages"
    img_exts = tuple(config.manifests.img_exts)
    seed = config.manifests.random_seed

    rows = []
    if shrimp_dir.exists():
        for cls_dir in sorted(shrimp_dir.iterdir()):
            if not cls_dir.is_dir():
                continue
            for fpath in cls_dir.iterdir():
                if fpath.suffix.lower() not in img_exts:
                    continue
                rows.append(
                    {
                        "image_path": str(fpath.resolve()),
                        "label": cls_dir.name,
                        "dataset": "shrimp",
                    }
                )

    df = pd.DataFrame(rows)
    if not df.empty:
        train_idx, temp_idx = train_test_split(
            df.index, test_size=0.30, stratify=df["label"], random_state=seed
        )
        temp_df = df.loc[temp_idx]
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=0.50,
            stratify=temp_df["label"],
            random_state=seed,
        )

        df["split"] = ""
        df.loc[train_idx, "split"] = "train"
        df.loc[val_idx, "split"] = "val"
        df.loc[test_idx, "split"] = "test"

    return df


def run(config):
    """Entry point: processes fish & shrimp datasets and writes manifest CSVs."""
    out_dir = config.paths.datasets / "manifests"
    out_dir.mkdir(parents=True, exist_ok=True)

    df_fish = build_fish_manifest(config)
    fish_out = out_dir / "fish_manifest.csv"
    df_fish.to_csv(fish_out, index=False)

    df_shrimp = build_shrimp_manifest(config)
    shrimp_out = out_dir / "shrimp_manifest.csv"
    df_shrimp.to_csv(shrimp_out, index=False)

    return {
        "fish_manifest_path": fish_out,
        "shrimp_manifest_path": shrimp_out,
        "fish_count": len(df_fish),
        "shrimp_count": len(df_shrimp),
    }


if __name__ == "__main__":
    from config import load_config

    cfg = load_config()
    res = run(cfg)
    print(
        f"Manifests generated successfully:\n - Fish: {res['fish_manifest_path']} ({res['fish_count']} samples)\n - Shrimp: {res['shrimp_manifest_path']} ({res['shrimp_count']} samples)"
    )
