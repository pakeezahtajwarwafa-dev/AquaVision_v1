import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

RESULTS_LOG_PATH = Path("outputs/experiments/model_comparison_log.csv")


def log_evaluation_result(
    species: str,
    backbone_name: str,
    fold_accs: list,
    fold_f1s: list,
    held_out_samples: int,
    checkpoint_dir: str = "checkpoints",
    confusion_matrix_path: str = None,
    notes: str = "",
) -> dict:
    """
    Append one row per (species, backbone) held-out evaluation run to a
    persistent, append-only CSV. This is the single source of truth for
    comparing architectures as you test CNN / EfficientNet / MobileViT /
    ViT / Swin variants -- without it, every run's numbers only exist as
    terminal output and get lost the moment the console scrolls.

    Safe to call repeatedly: each call adds one new row, nothing is
    overwritten, so the full history of every run stays intact.
    """
    RESULTS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    mean_acc = float(np.mean(fold_accs)) if fold_accs else None
    std_acc = float(np.std(fold_accs)) if fold_accs else None
    mean_f1 = float(np.mean(fold_f1s)) if fold_f1s else None
    std_f1 = float(np.std(fold_f1s)) if fold_f1s else None

    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "species": species,
        "backbone": backbone_name,
        "held_out_samples": held_out_samples,
        "n_folds_evaluated": len(fold_accs),
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc,
        "mean_macro_f1": mean_f1,
        "std_macro_f1": std_f1,
        "per_fold_accuracy": ";".join(f"{a:.4f}" for a in fold_accs),
        "per_fold_macro_f1": ";".join(f"{f:.4f}" for f in fold_f1s),
        "checkpoint_dir": checkpoint_dir,
        "confusion_matrix_path": confusion_matrix_path or "",
        "notes": notes,
    }

    if RESULTS_LOG_PATH.exists():
        existing = pd.read_csv(RESULTS_LOG_PATH)
        updated = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    else:
        updated = pd.DataFrame([row])

    updated.to_csv(RESULTS_LOG_PATH, index=False)

    f1_str = f"{mean_f1:.4f}" if mean_f1 is not None else "N/A"
    print(f"Logged result -> {RESULTS_LOG_PATH} | {species}/{backbone_name}: Mean Macro F1 {f1_str}")
    return row


def load_results_log() -> pd.DataFrame:
    """Convenience loader for comparing all logged runs so far."""
    if not RESULTS_LOG_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(RESULTS_LOG_PATH)
