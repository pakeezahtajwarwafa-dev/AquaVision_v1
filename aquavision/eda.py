import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image


def generate_class_distribution(df: pd.DataFrame, dataset_name: str, out_dir: Path) -> Path:
    plt.figure(figsize=(10, 5))
    ax = sns.countplot(data=df, x='label', hue='split', palette='viridis')
    plt.title(f'{dataset_name.capitalize()} Class Distribution by Split')
    plt.xlabel('Class Label')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    save_path = out_dir / f'{dataset_name}_class_distribution.png'
    plt.savefig(save_path, dpi=300)
    plt.close()
    return save_path


def inspect_image_properties(df: pd.DataFrame, dataset_name: str, out_dir: Path):
    widths, heights, aspect_ratios = [], [], []
    for path in df['image_path']:
        try:
            with Image.open(path) as img:
                w, h = img.size
                widths.append(w)
                heights.append(h)
                aspect_ratios.append(w / h)
        except Exception:
            continue

    df_props = pd.DataFrame({'width': widths, 'height': heights, 'aspect_ratio': aspect_ratios})

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.scatterplot(data=df_props, x='width', y='height', alpha=0.5, ax=axes[0], color='teal')
    axes[0].set_title(f'{dataset_name.capitalize()} Image Dimensions (WxH)')

    sns.histplot(df_props['aspect_ratio'], bins=20, kde=True, ax=axes[1], color='coral')
    axes[1].set_title(f'{dataset_name.capitalize()} Aspect Ratio Distribution')

    plt.tight_layout()
    save_path = out_dir / f'{dataset_name}_image_properties.png'
    plt.savefig(save_path, dpi=300)
    plt.close()
    return save_path, df_props.describe().to_dict()


def run(config):
    """Entry point: processes clean manifests and outputs EDA charts and summary statistics."""
    manifest_dir = config.paths.datasets / "manifests"
    out_dir = config.paths.outputs / "eda"
    out_dir.mkdir(parents=True, exist_ok=True)

    fish_in = manifest_dir / "fish_manifest_deduped.csv"
    shrimp_in = manifest_dir / "shrimp_manifest_deduped.csv"

    results = {}

    if fish_in.exists():
        df_fish = pd.read_csv(fish_in)
        fish_dist_path = generate_class_distribution(df_fish, "fish", out_dir)
        fish_props_path, fish_stats = inspect_image_properties(df_fish, "fish", out_dir)
        results["fish"] = {
            "count": len(df_fish),
            "class_dist_plot": fish_dist_path,
            "image_props_plot": fish_props_path,
            "stats": fish_stats
        }

    if shrimp_in.exists():
        df_shrimp = pd.read_csv(shrimp_in)
        shrimp_dist_path = generate_class_distribution(df_shrimp, "shrimp", out_dir)
        shrimp_props_path, shrimp_stats = inspect_image_properties(df_shrimp, "shrimp", out_dir)
        results["shrimp"] = {
            "count": len(df_shrimp),
            "class_dist_plot": shrimp_dist_path,
            "image_props_plot": shrimp_props_path,
            "stats": shrimp_stats
        }

    return results


if __name__ == "__main__":
    from config import load_config

    cfg = load_config()
    res = run(cfg)
    print("EDA Complete. Figures saved to outputs/eda/")
