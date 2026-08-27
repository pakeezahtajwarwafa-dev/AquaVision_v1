import cv2
import pandas as pd
from pathlib import Path
from PIL import Image, ImageOps

def verify_and_clean_manifest(manifest_path: Path) -> pd.DataFrame:
    """Checks image decodability and EXIF orientation across manifest records."""
    if not manifest_path.exists():
        return pd.DataFrame()

    df = pd.read_csv(manifest_path)
    valid_rows = []
    corrupt_count = 0

    for idx, row in df.iterrows():
        p = str(row["image_path"])
        try:
            # Step 1: Open with PIL to check EXIF rotation and header integrity
            with Image.open(p) as img:
                img = ImageOps.exif_transpose(img)
                img.verify()
            
            # Step 2: Decode pixels with OpenCV to ensure compatibility
            cv_img = cv2.imread(p)
            if cv_img is None or cv_img.size == 0:
                corrupt_count += 1
                continue
                
            valid_rows.append(row.to_dict())
        except Exception:
            corrupt_count += 1

    clean_df = pd.DataFrame(valid_rows)
    print(f"[{manifest_path.name}] Verified {len(clean_df)} valid images. (Dropped {corrupt_count} corrupt/unreadable files)")
    return clean_df

def run():
    manifest_dir = Path("datasets/manifests")
    for species in ["fish", "shrimp"]:
        manifest_file = manifest_dir / f"{species}_manifest_deduped.csv"
        if manifest_file.exists():
            df_clean = verify_and_clean_manifest(manifest_file)
            df_clean.to_csv(manifest_file, index=False)

if __name__ == "__main__":
    run()
