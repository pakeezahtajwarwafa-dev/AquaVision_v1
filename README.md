# AquaVision BD 🐟🦐

**A Multimodal Deep Learning System for Aquatic Disease Diagnosis in Bangladeshi Fish & Shrimp Farming**

AquaVision BD is a computer-vision + tabular-data diagnostic platform that helps identify diseases in farmed fish and shrimp. A user uploads a photo of the animal along with basic water quality readings (temperature, pH, dissolved oxygen, ammonia), and the system fuses both signals to predict the most likely health condition — while guarding against invalid or mismatched inputs and pointing to BFRI (Bangladesh Fisheries Research Institute) treatment guidance.

---

## 1. What the Project Does

1. **Image + water-quality fusion diagnosis** — a CNN backbone (ResNet18 / EfficientNet-B0, extensible to ViT/Swin/MobileViT) extracts visual features from the uploaded photo, which are concatenated with engineered water-quality features and passed through a fusion classifier head.
2. **Species & out-of-distribution (OOD) gating** — before diagnosis runs, a separate ResNet18 "gate" model checks whether the image actually shows a fish, a shrimp, or something else (`other`), and whether it matches the species the user selected. Mismatched or non-aquatic images are rejected with a clear warning instead of producing a misleading diagnosis.
3. **Explainability** — Grad-CAM (`explainability.py`) generates class activation maps so predictions can be visually inspected rather than trusted blindly.
4. **BFRI treatment lookup (RAG)** — a lightweight retrieval engine (`aquavision/rag/bfri_rag.py`) maps a predicted disease to symptoms and BFRI-recommended treatment/water-remediation steps.
5. **Research/analytics dashboard** — a FastAPI backend serves a live admin dashboard (dataset stats, model metrics, live submission gallery) in addition to the prediction API.
6. **Streamlit demo app** (`app.py`) — a simple UI for uploading an image, setting water-quality sliders, and viewing the diagnosis.

---

## 2. Project Structure

```
AquaVision_v1/
├── app.py                     # Streamlit demo dashboard (image + WQ sliders → diagnosis)
├── config.py / config.yaml    # Global paths, training, and ablation configuration
├── requirements.txt           # Python dependencies
│
├── aquavision/                 # Core package
│   ├── engine/
│   │   └── predictor.py        # AquaPredictor: loads models, runs species gate + diagnosis
│   ├── fusion.py                # AquaVisionMultimodalModel (image backbone + tabular fusion head)
│   ├── models.py / models/vision_backbone.py
│   ├── train.py, train_gate.py  # Training loops for diagnostic model & species/OOD gate
│   ├── evaluate.py              # Cross-validation & holdout evaluation
│   ├── explainability.py        # Grad-CAM class activation maps
│   ├── water_quality.py         # Synthetic water-quality data generation + tabular model
│   ├── dataset.py, data_prep.py, preprocessing.py  # Dataset construction & image preprocessing
│   ├── dedupe.py, manifests.py, rebuild_manifests.py  # Near-duplicate detection (dHash) & manifest building
│   ├── eda.py                   # Exploratory data analysis
│   ├── export_onnx.py           # Export trained models to ONNX
│   ├── audit.py, results_log.py, live_log.py  # Logging & auditing utilities
│   ├── rag/bfri_rag.py          # BFRI disease → treatment retrieval engine
│   └── api/
│       ├── main.py              # FastAPI app: /predict-multimodal, /health, /test UI
│       └── admin/                # Analytics dashboard (router, views, engine)
│
├── datasets/                    # Raw & processed fish/shrimp/water-quality datasets, manifests, EDA reports
├── checkpoints/                 # Trained model weights (per species, per backbone, per CV fold) + scalers
├── exports/                     # ONNX-exported models
├── notebooks/                    # 01–08: manifests → dedup → EDA → dataset/model → training → evaluation → water quality → fusion
├── outputs/                      # EDA plots, experiment logs, live submission history
└── archive_legacy/                # Earlier/legacy pipeline scripts kept for reference
```

The `notebooks/` folder documents the end-to-end pipeline in order:
`01_build_manifests → 02_check_near_duplicates → 03_eda_and_quality → 04_dataset_and_models → 05_train_and_ablation → 06_evaluate_and_explain → 07_water_quality_analysis → 08_multimodal_fusion`

---

## 3. Data

| Species | Classes | Samples (processed manifest) |
|---|---|---|
| **Fish** | Bacterial Red disease, Bacterial diseases – Aeromoniasis, Bacterial gill disease, EUS, Fungal diseases – Saprolegniasis, Healthy Fish, Parasitic diseases, Viral diseases – White tail disease | 2,731 |
| **Shrimp** | BG, Healthy, WSSV, WSSV_BG | 1,506 |

Water-quality readings (temperature, pH, dissolved oxygen, ammonia, plus engineered features like `temp_do_ratio`, `ammonia_toxicity_index`, `stress_score`) are synthesized and correlated with disease labels (`water_quality.py`) with controlled label noise for realism. Near-duplicate images are removed via perceptual (dHash) hashing to prevent train/test leakage (`dedupe.py`).

---

## 4. Models & Results

- **Diagnostic model:** `AquaVisionMultimodalModel` — a swappable vision backbone (ResNet18 by default; EfficientNet-B0, ViT, Swin, and MobileViT also supported) with global average pooled features concatenated with scaled tabular water-quality features, fed into a `Linear → BatchNorm → ReLU → Dropout → Linear` classifier head.
- **Species/OOD gate:** a ResNet18 3-way classifier (`fish` / `shrimp` / `other`) used to reject invalid uploads and species mismatches before diagnosis.
- **Explainability:** Grad-CAM over the backbone's final convolutional layer.

**5-fold cross-validation (ResNet18, held-out test set):**

| Species | Mean Accuracy | Mean Macro-F1 | Std Dev (F1) |
|---|---|---|---|
| Fish | 93.9% | 0.940 | 0.008 |
| Shrimp | 83.2% | 0.832 | 0.016 |

Both species are flagged `STABLE` (low fold-to-fold variance) in `datasets/cv_metrics.csv`. Per-fold results, confusion matrices, and additional backbone comparisons are logged in `outputs/experiments/model_comparison_log.csv` and `datasets/*_confusion_matrix.csv`.

Trained weights for each species/backbone/fold combination, plus the corresponding feature scalers, live in `checkpoints/`, tracked centrally by `checkpoints/production_model_registry.json` (which backbone + fold is "in production" per species).

---

## 5. How Prediction Works (`AquaPredictor`)

1. Load the production backbone/fold for the requested species from the registry, along with the matching manifest (for class names), scaler, and model checkpoint.
2. Preprocess the uploaded image (letterbox resize to 224×224, ImageNet normalization).
3. Run the **species/OOD gate** first:
   - If the top class is `other` or confidence is low → reject as *Invalid Subject*.
   - If the gate detects a different species than selected (with reasonable confidence) → reject as *Species Mismatch*.
4. If the image passes the gate, engineer the tabular water-quality features, scale them, and run the fused **diagnostic model** to produce a disease prediction with a full probability breakdown.

---

## 6. Running the Project

### Setup
```bash
pip install -r requirements.txt
```

### Streamlit demo
```bash
streamlit run app.py
```
Upload a fish/shrimp image, set the water-quality sliders, and click **Run Diagnosis**.

### FastAPI backend + dashboard
```bash
uvicorn aquavision.api.main:app --reload
```
- `GET /` → redirects to the research/analytics dashboard (`/admin/dashboard`)
- `GET /admin/api/metrics`, `GET /admin/api/live-metrics` → dataset/model/live-submission JSON stats
- `POST /predict-multimodal` → multipart form (`file`, `species`, `water_quality` JSON string) → diagnosis JSON
- `GET /test` → a self-contained HTML testing UI for the prediction endpoint
- `GET /health` → service health check

### Training / evaluation (advanced)
- `aquavision/train.py` — trains the multimodal diagnostic model
- `aquavision/train_gate.py` — trains the species/OOD gate
- `aquavision/evaluate.py` — runs cross-validation / holdout evaluation
- `aquavision/export_onnx.py` — exports a trained checkpoint to ONNX for lightweight inference

---

## 7. Tech Stack

Python · PyTorch / torchvision · timm · Albumentations · OpenCV · scikit-learn · pandas / numpy · FastAPI · Streamlit · ONNX · SHAP · Optuna

---

## 8. Notes

- This is an academic (course project) build — the water-quality dataset is **synthetically generated** and correlated with labels, not sourced from live IoT sensors.
- `archive_legacy/` and files suffixed `_legacy` reflect earlier iterations of the pipeline kept for reference/comparison as the project evolved (dataset dedup, multi-backbone support, and the species gate were added in later milestones — see commit history).
