import os
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, train_test_split

def preprocess_datasets():
    base_path = Path(__file__).resolve().parents[2]
    datasets_dir = base_path / "datasets"
    processed_dir = datasets_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Exclude system folders
    ignored_folders = {"processed", "manifests", "bfri_docs", "eda_outputs", "__pycache__"}
    
    species_folders = [
        f for f in datasets_dir.iterdir() 
        if f.is_dir() and f.name not in ignored_folders and not f.name.startswith(".")
    ]
    
    if not species_folders:
        print(f"\n[ERROR] No raw dataset folders found inside {datasets_dir}")
        return

    for species_dir in species_folders:
        species_name = species_dir.name
        print(f"\n--- Processing Dataset: {species_name.upper()} ---")
        
        data = []
        class_folders = [f for f in species_dir.iterdir() if f.is_dir()]
        
        for class_folder in class_folders:
            class_name = class_folder.name
            images = []
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG"]:
                images.extend(list(class_folder.glob(ext)))
                
            for img_path in images:
                data.append({
                    "image_path": str(img_path.relative_to(base_path)).replace("\\", "/"),
                    "label": class_name,
                    "is_augmented": 0
                })
                
        if not data:
            print(f"  -> No images found for {species_name}. Skipping.")
            continue
            
        df = pd.DataFrame(data)
        print(f"  -> Found {len(df)} images across {df['label'].nunique()} classes.")
        
        df['fold'] = -1
        try:
            train_idx, test_idx = train_test_split(
                df.index, test_size=0.1, random_state=42, stratify=df['label']
            )
            df.loc[test_idx, 'fold'] = -1
            
            train_df = df.loc[train_idx].copy()
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            
            for fold, (_, val_idx) in enumerate(skf.split(train_df, train_df['label'])):
                actual_val_idx = train_df.iloc[val_idx].index
                df.loc[actual_val_idx, 'fold'] = fold
                
        except ValueError:
            print(f"  -> Defaulting to simple fold assignment.")
            df['fold'] = df.index % 5
            
        out_path = processed_dir / f"{species_name}_processed_manifest.csv"
        df.to_csv(out_path, index=False)
        print(f"  -> Success! Generated manifest: {out_path}")

if __name__ == "__main__":
    preprocess_datasets()
