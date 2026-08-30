import torch
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

from aquavision.dataset import AquaDataset
from aquavision.fusion import AquaVisionMultimodalModel
from aquavision.results_log import log_evaluation_result


def get_confusion_matrix_df(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred)
    return pd.DataFrame(cm, index=labels, columns=labels)


def _run_inference(model, test_loader, device):
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 3:
                images, tabular, labels = batch
                images, tabular = images.to(device), tabular.to(device)
                outputs = model(images, tabular)
            else:
                images, labels = batch
                images = images.to(device)
                outputs = model(images)

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    return all_preds, all_labels


def evaluate_held_out(species: str = "fish", backbone_name: str = "resnet18", checkpoint_prefix: str = None, notes: str = ""):
    """
    True zero-leakage evaluation: filters to fold == -1 (the held-out set
    carved out by rebuild_manifests.py, either from a real test_split folder
    or a stratified carve-out), which no fold's training/validation ever
    touches. Evaluates every trained fold checkpoint against this SAME fixed
    set and reports the average -- a far more robust final number than any
    single fold's CV validation score.

    backbone_name / checkpoint_prefix let this be reused for future
    architecture comparisons (EfficientNet, MobileViT, ViT, Swin, ...)
    without editing this function -- checkpoint_prefix defaults to today's
    naming convention ({species}_best_model), override it once train.py
    starts saving per-backbone checkpoints (e.g. {species}_{backbone}_best_model).
    Every call appends its result to outputs/experiments/model_comparison_log.csv.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=== ZERO-LEAKAGE HELD-OUT EVALUATION: {species.upper()} ({backbone_name}) ===")

    checkpoint_prefix = checkpoint_prefix or f"{species}_best_model"

    processed_manifest = Path("datasets/processed") / f"{species}_processed_manifest.csv"
    if not processed_manifest.exists():
        print(f"Manifest missing for {species}")
        return

    df = pd.read_csv(processed_manifest)
    test_df = df[df["fold"] == -1].copy()

    if len(test_df) == 0:
        print(f"[ERROR] No held-out test samples found for {species} (fold == -1).")
        return

    unique_labels = sorted(df["label"].unique())
    label2idx = {lbl: idx for idx, lbl in enumerate(unique_labels)}
    test_df["label_idx"] = test_df["label"].map(label2idx)

    tab_cols = [c for c in ["temperature", "ph", "dissolved_oxygen", "ammonia", "temp_do_ratio", "ammonia_toxicity_index", "stress_score"] if c in df.columns]

    checkpoints_dir = Path("checkpoints")
    fold_accs, fold_f1s = [], []
    fold_predictions = {}  # fold -> (all_preds, all_labels), reused below for the detailed report

    for fold in range(5):
        checkpoint_path = checkpoints_dir / f"{checkpoint_prefix}_fold{fold}.pt"
        if not checkpoint_path.exists():
            print(f"  > Fold {fold}: checkpoint not found, skipping ({checkpoint_path})")
            continue

        fold_test_df = test_df.copy()

        # Use the SAME scaler that fold's model was trained with, rather than
        # refitting -- matches how the live predictor applies scaling.
        if tab_cols:
            scaler_path = checkpoints_dir / f"{species}_scaler_fold{fold}.pkl"
            if scaler_path.exists():
                scaler = joblib.load(scaler_path)
                fold_test_df[tab_cols] = scaler.transform(fold_test_df[tab_cols].fillna(0.0))
            else:
                fold_test_df[tab_cols] = fold_test_df[tab_cols].fillna(0.0)

        test_ds = AquaDataset(fold_test_df, img_size=224, is_train=False, species=species, tab_cols=tab_cols)
        test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

        model = AquaVisionMultimodalModel(
            num_classes=len(unique_labels),
            num_tabular_features=len(tab_cols),
            backbone_name=backbone_name,
            pretrained=False  # we load a full trained checkpoint next line, no need to fetch ImageNet weights first
        ).to(device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
        model.eval()

        all_preds, all_labels = _run_inference(model, test_loader, device)
        fold_predictions[fold] = (all_preds, all_labels)

        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="macro")
        fold_accs.append(acc)
        fold_f1s.append(f1)
        print(f"  > Fold {fold} model on held-out set -> Acc: {acc:.4f} | Macro F1: {f1:.4f}")

    if not fold_f1s:
        print(f"[ERROR] No checkpoints found for {species}, nothing evaluated.")
        return

    mean_acc, std_acc = float(np.mean(fold_accs)), float(np.std(fold_accs))
    mean_f1, std_f1 = float(np.mean(fold_f1s)), float(np.std(fold_f1s))
    print(f"\n[{species.upper()} ZERO-LEAKAGE SUMMARY] Held-out samples: {len(test_df)}")
    print(f"  Mean Accuracy: {mean_acc:.4f} (+/-{std_acc:.4f})")
    print(f"  Mean Macro F1: {mean_f1:.4f} (+/-{std_f1:.4f})")

    # Detailed report + confusion matrix for whichever fold scored best on
    # THIS held-out set (reuses predictions already computed above).
    evaluated_folds = list(fold_predictions.keys())
    best_fold = evaluated_folds[int(np.argmax(fold_f1s))]
    all_preds, all_labels = fold_predictions[best_fold]
    print(f"\nDetailed report below uses Fold {best_fold} (best-performing on this held-out set):")

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=unique_labels, digits=4, zero_division=0))

    cm_df = get_confusion_matrix_df(all_labels, all_preds, unique_labels)
    cm_out = Path("datasets") / f"{species}_{backbone_name}_holdout_confusion_matrix.csv"
    cm_df.to_csv(cm_out)
    print(f"Saved confusion matrix to: {cm_out}\n" + "=" * 50 + "\n")

    log_evaluation_result(
        species=species,
        backbone_name=backbone_name,
        fold_accs=fold_accs,
        fold_f1s=fold_f1s,
        held_out_samples=len(test_df),
        checkpoint_dir=str(checkpoints_dir),
        confusion_matrix_path=str(cm_out),
        notes=notes,
    )


def run():
    for species in ["fish", "shrimp"]:
        evaluate_held_out(species)


if __name__ == "__main__":
    run()
