import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, Any

class DeveloperAnalyticsEngine:
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.processed_dir = self.base_dir / "datasets" / "processed"
        self.checkpoints_dir = self.base_dir / "checkpoints"
        self.exports_dir = self.base_dir / "exports"
        self.bfri_path = self.base_dir / "datasets" / "bfri_docs" / "bfri_guidelines.json"

    def get_manifest_stats(self, species: str) -> Dict[str, Any]:
        manifest_path = self.processed_dir / f"{species}_processed_manifest.csv"
        if not manifest_path.exists():
            return {"status": "missing", "message": f"Manifest not found at {manifest_path}"}

        df = pd.read_csv(manifest_path)

        total_images = len(df)
        heldout_test_count = int((df["fold"] == -1).sum()) if "fold" in df.columns else 0
        cv_train_val_count = total_images - heldout_test_count
        augmented_count = int(df["is_augmented"].sum()) if "is_augmented" in df.columns else 0

        fold_counts = df["fold"].value_counts().to_dict() if "fold" in df.columns else {}
        class_counts = df["label"].value_counts().to_dict()

        tab_cols = [c for c in ["temperature", "ph", "dissolved_oxygen", "ammonia", "temp_do_ratio", "ammonia_toxicity_index", "stress_score"] if c in df.columns]
        wq_summary = {}
        for col in tab_cols:
            valid_series = df[col].dropna()
            if not valid_series.empty:
                wq_summary[col] = {
                    "mean": round(float(valid_series.mean()), 3),
                    "min": round(float(valid_series.min()), 3),
                    "max": round(float(valid_series.max()), 3),
                    "missing_count": int(df[col].isna().sum())
                }

        return {
            "status": "active",
            "total_images": total_images,
            "cv_pool_size": cv_train_val_count,
            "heldout_test_size": heldout_test_count,
            "augmented_samples": augmented_count,
            "fold_distribution": {f"fold_{k}": v for k, v in fold_counts.items()},
            "class_distribution": class_counts,
            "water_quality_analytics": wq_summary
        }

    def get_system_summary(self) -> Dict[str, Any]:
        fish_stats = self.get_manifest_stats("fish")
        shrimp_stats = self.get_manifest_stats("shrimp")

        pt_files = [f.name for f in self.checkpoints_dir.glob("*.pt")] if self.checkpoints_dir.exists() else []
        scaler_files = [f.name for f in self.checkpoints_dir.glob("*.pkl")] if self.checkpoints_dir.exists() else []
        onnx_files = [f.name for f in self.exports_dir.glob("*.onnx")] if self.exports_dir.exists() else []

        bfri_count = 0
        if self.bfri_path.exists():
            with open(self.bfri_path, "r", encoding="utf-8-sig") as f:
                bfri_docs = json.load(f)
                bfri_count = len(bfri_docs)

        return {
            "dataset_summary": {
                "fish": fish_stats,
                "shrimp": shrimp_stats
            },
            "system_artifacts": {
                "pytorch_checkpoints": pt_files,
                "scaler_files": scaler_files,
                "onnx_models": onnx_files,
                "bfri_protocols_count": bfri_count
            }
        }
