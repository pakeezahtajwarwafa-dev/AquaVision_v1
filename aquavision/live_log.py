import json
import uuid
from pathlib import Path
from datetime import datetime, timezone

import cv2
import numpy as np
import pandas as pd

# Deliberately kept separate from datasets/processed/*.csv (the labeled
# training manifests). Live submissions are unverified model predictions,
# not ground truth -- they must never silently blend into training/holdout
# counts. This is its own append-only stream for usage analytics.
LIVE_LOG_DIR = Path("outputs/live_submissions")
THUMBNAILS_DIR = LIVE_LOG_DIR / "thumbnails"
LIVE_LOG_PATH = LIVE_LOG_DIR / "live_predictions_log.csv"
THUMBNAIL_MAX_SIDE = 160


def _save_thumbnail(image_rgb: np.ndarray, submission_id: str) -> str:
    THUMBNAILS_DIR.mkdir(parents=True, exist_ok=True)
    h, w = image_rgb.shape[:2]
    scale = THUMBNAIL_MAX_SIDE / max(h, w)
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    thumb = cv2.resize(image_rgb, new_size, interpolation=cv2.INTER_AREA)
    thumb_bgr = cv2.cvtColor(thumb, cv2.COLOR_RGB2BGR)
    filename = f"{submission_id}.jpg"
    cv2.imwrite(str(THUMBNAILS_DIR / filename), thumb_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return filename


def log_live_submission(image_rgb: np.ndarray, species: str, result: dict, water_quality: dict = None) -> dict:
    """
    Records one /test (or API) submission: a thumbnail plus prediction
    metadata. Never raises -- logging failures must not break the actual
    prediction response, so callers should still wrap this in try/except.
    """
    LIVE_LOG_DIR.mkdir(parents=True, exist_ok=True)
    submission_id = uuid.uuid4().hex[:12]
    thumb_filename = _save_thumbnail(image_rgb, submission_id)

    is_mismatch = bool(result.get("is_mismatch", False))
    row = {
        "submission_id": submission_id,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "species": species,
        "is_mismatch": is_mismatch,
        "prediction": "" if is_mismatch else result.get("prediction", ""),
        "confidence": None if is_mismatch else result.get("confidence", None),
        "gate_message": result.get("message", "") if is_mismatch else "",
        "thumbnail_filename": thumb_filename,
        "water_quality_json": json.dumps(water_quality or {}),
        "all_probabilities_json": json.dumps(result.get("all_probabilities", {})),
    }

    if LIVE_LOG_PATH.exists():
        existing = pd.read_csv(LIVE_LOG_PATH)
        updated = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
    else:
        updated = pd.DataFrame([row])
    updated.to_csv(LIVE_LOG_PATH, index=False)
    return row


def load_live_log() -> pd.DataFrame:
    if not LIVE_LOG_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(LIVE_LOG_PATH)
