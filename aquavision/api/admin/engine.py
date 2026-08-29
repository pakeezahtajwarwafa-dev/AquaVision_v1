import pandas as pd
from pathlib import Path
from typing import Dict, Any, List

class DynamicAnalyticsEngine:
    def __init__(self):
        # Strictly use the terminal's working directory
        self.base_dir = Path.cwd()
        self.processed_dir = self.base_dir / "datasets" / "processed"
        self.checkpoints_dir = self.base_dir / "checkpoints"
        self.exports_dir = self.base_dir / "exports"
        
        # Debug outputs to your terminal
        print(f"\n[SYSTEM] Looking for datasets in: {self.processed_dir}")
        print(f"[SYSTEM] Directory exists? {self.processed_dir.exists()}")

    def get_available_datasets(self) -> List[str]:
        if not self.processed_dir.exists():
            return []
        
        datasets = []
        # Find all CSV files in the directory
        for file in self.processed_dir.glob("*.csv"):
            if "processed_manifest" in file.name:
                name = file.name.replace("_processed_manifest.csv", "")
                datasets.append(name)
                
        print(f"[SYSTEM] Found datasets: {datasets}\n")
        return sorted(datasets)

    def get_dataset_metrics(self) -> Dict[str, Any]:
        datasets = self.get_available_datasets()
        metrics = {}
        total_global_samples = 0
        total_global_heldout = 0

        for ds in datasets:
            manifest_path = self.processed_dir / f"{ds}_processed_manifest.csv"
            df = pd.read_csv(manifest_path)
            
            total = len(df)
            total_global_samples += total
            
            heldout = int((df["fold"] == -1).sum()) if "fold" in df.columns else 0
            total_global_heldout += heldout
            
            augmented = int(df.get("is_augmented", pd.Series([0])).sum()) if "is_augmented" in df.columns else 0
            
            fold_dist = {}
            if "fold" in df.columns:
                for f in sorted(df["fold"].unique()):
                    f_df = df[df["fold"] == f]
                    fold_dist[f"Fold {f}"] = {
                        "original": int((f_df.get("is_augmented", 0) == 0).sum()),
                        "augmented": int((f_df.get("is_augmented", 0) == 1).sum())
                    }
                    
            # Fallback if label column is missing
            label_col = "label" if "label" in df.columns else df.columns[1] 
            
            metrics[ds] = {
                "total": total,
                "heldout": heldout,
                "cv_pool": total - heldout,
                "augmented_ratio": round(augmented / total * 100, 1) if total > 0 else 0,
                "class_balance": df[label_col].value_counts().to_dict(),
                "fold_breakdown": fold_dist
            }
            
        return {
            "datasets": metrics,
            "global_stats": {
                "total_samples": total_global_samples,
                "total_heldout": total_global_heldout,
                "dataset_count": len(datasets)
            }
        }

    def get_model_metrics(self) -> Dict[str, Any]:
        artifacts = {"pytorch_checkpoints": {}, "onnx_exports": {}}
        
        if self.checkpoints_dir.exists():
            for pt in self.checkpoints_dir.glob("*.pt"):
                arch = pt.stem.split("_")[0] if "_" in pt.stem else "custom_model"
                if arch not in artifacts["pytorch_checkpoints"]:
                    artifacts["pytorch_checkpoints"][arch] = []
                artifacts["pytorch_checkpoints"][arch].append(pt.name)
                
        if self.exports_dir.exists():
            for onnx_file in self.exports_dir.glob("*.onnx"):
                arch = onnx_file.stem.split("_")[0] if "_" in onnx_file.stem else "custom_model"
                if arch not in artifacts["onnx_exports"]:
                    artifacts["onnx_exports"][arch] = []
                artifacts["onnx_exports"][arch].append(onnx_file.name)
                
        return artifacts
