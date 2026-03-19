"""
Model loading service.

SECURITY: Only .joblib files are accepted. Every load verifies the
SHA-256 hash stored at upload time. If the file is missing or the
hash does not match, we raise — we NEVER silently substitute a
dummy model, which would produce meaningless audit results.
"""
from __future__ import annotations
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from app.utils.security import verify_file_hash

logger = logging.getLogger(__name__)


def load_verified_model(model_record):
    """
    Load a model from disk, verifying its SHA-256 hash before deserialisation.

    Raises:
        ValueError: if metadata is missing, file not found, or hash mismatch.
    """
    meta = model_record.model_metadata or {}
    file_path = meta.get("file_path")
    expected_hash = meta.get("sha256")

    if not file_path or not expected_hash:
        raise ValueError(
            f"Model id={model_record.id} has no uploaded file. "
            "Upload a .joblib file via POST /api/models/upload before auditing."
        )

    path = Path(file_path)
    if not path.is_file():
        raise ValueError(
            f"Model file not found at '{file_path}'. "
            "The file may have been deleted. Re-upload the model."
        )

    if not verify_file_hash(path, expected_hash):
        raise ValueError(
            f"SHA-256 mismatch for model id={model_record.id}. "
            "The file has been modified since upload and cannot be trusted."
        )

    logger.info("Loading model id=%d from %s (hash OK)", model_record.id, file_path)
    return joblib.load(str(path))


def evaluate_model(model, X: pd.DataFrame, y: pd.Series, protected: pd.Series) -> dict:
    """Return overall and per-group classification metrics."""
    y_pred = model.predict(X)
    overall = {
        "accuracy": float(accuracy_score(y, y_pred)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall": float(recall_score(y, y_pred, zero_division=0)),
        "f1": float(f1_score(y, y_pred, zero_division=0)),
    }
    group_metrics: dict = {}
    for g in sorted(protected.unique()):
        mask = protected == g
        if mask.sum() > 0:
            group_metrics[str(g)] = {
                "accuracy": float(accuracy_score(y[mask], y_pred[mask])),
                "precision": float(precision_score(y[mask], y_pred[mask], zero_division=0)),
                "recall": float(recall_score(y[mask], y_pred[mask], zero_division=0)),
                "f1": float(f1_score(y[mask], y_pred[mask], zero_division=0)),
                "count": int(mask.sum()),
            }
    return {"overall": overall, "by_group": group_metrics}
