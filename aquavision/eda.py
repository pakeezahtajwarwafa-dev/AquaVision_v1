import pandas as pd
from pathlib import Path
from config import load_config

def generate_eda_report(species: str, manifest_dir: Path, output_dir: Path):
    raw_manifest = manifest_dir / f"{species}_manifest.csv"
    dedup_manifest = manifest_dir / f"{species}_manifest_deduped.csv"

    if not dedup_manifest.exists():
        print(f"Manifest missing: {dedup_manifest}")
        return

    df_clean = pd.read_csv(dedup_manifest)
    
    # 1. Compute cleaned dataset summary
    summary = df_clean.groupby('label').agg(
        clean_count=('image_path', 'count'),
        augmented_count=('is_augmented', lambda x: x.sum() if 'is_augmented' in df_clean.columns else 0)
    ).reset_index()

    summary['aug_percent'] = (summary['augmented_count'] / summary['clean_count'] * 100).round(1)

    # 2. Merge with raw baseline count if original manifest exists
    if raw_manifest.exists():
        df_raw = pd.read_csv(raw_manifest)
        raw_counts = df_raw.groupby('label')['image_path'].count().reset_index()
        raw_counts.columns = ['label', 'original_count']
        summary = pd.merge(raw_counts, summary, on='label', how='right')
        summary['removed_duplicates'] = summary['original_count'] - summary['clean_count']

    # 3. Save as a separate NEW report file (does not overwrite baseline)
    output_file = output_dir / f"{species}_eda_report_cleaned.csv"
    summary.to_csv(output_file, index=False)

    # 4. Print comparative summary
    print(f"=== {species.upper()} DATASET EDA COMPARISON ===")
    print(f"Saved New Report: {output_file}\n")
    print(summary.to_string(index=False))
    print("\nSplit Distribution (Cleaned):")
    print(df_clean['split'].value_counts().to_string())
    print("\n" + "="*50 + "\n")

def run(config):
    manifest_dir = Path("datasets/manifests")
    output_dir = Path("datasets")
    output_dir.mkdir(parents=True, exist_ok=True)

    for species in ["fish", "shrimp"]:
        generate_eda_report(species, manifest_dir, output_dir)

if __name__ == "__main__":
    cfg = load_config()
    run(cfg)
