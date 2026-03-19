"""
Audit execution engine.

Computes three categories of ethics metrics:
  - Fairness  : demographic parity, equalized odds, disparate impact
  - Explainability : SHAP-based feature importance, model complexity
  - Robustness: prediction stability under Gaussian noise

No dummy data, no fallback models. If data or model is missing, the
audit fails with a clear error message.
"""
from __future__ import annotations
import logging
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker

try:
    import shap
    _SHAP = True
except ImportError:
    _SHAP = False

try:
    from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
    _FAIRLEARN = True
except ImportError:
    _FAIRLEARN = False

from app.models.audit import Audit, AuditMetric
from app.services.model_service import load_verified_model

logger = logging.getLogger(__name__)

_REQUIRED_COLS = {"target", "protected_attribute"}


def _get_session():
    db_url = os.environ.get("DATABASE_URL", "sqlite:///data/app.db")
    engine = create_engine(db_url, connect_args={"check_same_thread": False})
    return scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))


def _load_data(model_record) -> pd.DataFrame:
    """
    Locate and load the test dataset for this model's domain.
    Raises ValueError if no data file can be found.
    """
    domain = (model_record.model_metadata or {}).get("domain", "")
    candidates = []
    if domain:
        candidates.append(Path(f"data/{domain}_data.csv"))
    candidates.append(Path("data/testing_data.csv"))

    for path in candidates:
        if path.is_file():
            df = pd.read_csv(path)
            missing = _REQUIRED_COLS - set(df.columns)
            if missing:
                raise ValueError(
                    f"Dataset '{path}' is missing required columns: {missing}. "
                    "Columns 'target' and 'protected_attribute' are mandatory."
                )
            logger.info("Loaded dataset: %s (%d rows)", path, len(df))
            return df

    raise ValueError(
        "No test dataset found. Place a CSV with 'target' and 'protected_attribute' "
        "columns at data/testing_data.csv before running an audit."
    )


def run_audit(audit_id: int) -> None:
    """Execute a full ethics audit and persist metrics."""
    db = _get_session()
    try:
        audit = db.query(Audit).get(audit_id)
        if not audit:
            raise ValueError(f"Audit id={audit_id} not found.")

        audit.status = "running"
        audit.updated_at = datetime.utcnow()
        db.commit()

        # --- load model (raises on missing file or hash mismatch) ---
        model = load_verified_model(audit.model)

        # --- load data (raises if missing or malformed) ---
        df = _load_data(audit.model)
        X = df.drop(list(_REQUIRED_COLS), axis=1)
        y = df["target"]
        protected = df["protected_attribute"]

        # --- compute metrics ---
        metrics: list[dict] = []
        metrics.extend(_fairness(model, X, y, protected))
        metrics.extend(_explainability(model, X))
        metrics.extend(_robustness(model, X, y))

        for m in metrics:
            db.add(AuditMetric(
                audit_id=audit.id,
                name=m["name"],
                category=m["category"],
                value=m["value"],
                details=m["details"],
            ))

        audit.status = "completed"
        audit.updated_at = datetime.utcnow()
        db.commit()
        logger.info("Audit id=%d completed with %d metrics.", audit_id, len(metrics))

    except Exception as exc:
        logger.error("Audit id=%d failed: %s", audit_id, exc)
        db.query(Audit).filter_by(id=audit_id).update({"status": "failed"})
        db.commit()
        raise
    finally:
        db.remove()


# ────────────────────────────── Fairness ──────────────────────────────────────

def _fairness(model, X: pd.DataFrame, y: pd.Series, protected: pd.Series) -> list[dict]:
    y_pred = model.predict(X)
    groups = sorted(protected.unique())

    group_rates = {}
    for g in groups:
        mask = protected == g
        group_rates[str(g)] = float(y_pred[mask].mean()) if mask.any() else 0.0

    rate_vals = list(group_rates.values())
    stat_parity = float(max(rate_vals) - min(rate_vals))
    denom = max(rate_vals)
    disp_impact = float(min(rate_vals) / denom) if denom > 0 else 1.0

    if _FAIRLEARN:
        dp_diff = float(demographic_parity_difference(y_true=y, y_pred=y_pred, sensitive_features=protected))
        eo_diff = float(equalized_odds_difference(y_true=y, y_pred=y_pred, sensitive_features=protected))
    else:
        dp_diff = stat_parity
        eo_diff = stat_parity
        logger.warning("fairlearn not installed — using manual fairness approximations.")

    return [
        {"name": "Demographic Parity Difference", "category": "fairness", "value": dp_diff,
         "details": {"description": "Difference in selection rates between groups.",
                     "interpretation": "0 = perfect parity. >0.1 signals concern.",
                     "group_rates": group_rates, "library": "fairlearn" if _FAIRLEARN else "manual"}},
        {"name": "Equalized Odds Difference", "category": "fairness", "value": eo_diff,
         "details": {"description": "Max delta of TPR and FPR between groups.",
                     "interpretation": "0 = equalized odds.", "library": "fairlearn" if _FAIRLEARN else "manual"}},
        {"name": "Disparate Impact Ratio", "category": "fairness", "value": disp_impact,
         "details": {"description": "Ratio of the lowest to highest group selection rate.",
                     "interpretation": "1 = equal impact. Below 0.8 violates the 80% rule."}},
        {"name": "Statistical Parity", "category": "fairness", "value": stat_parity,
         "details": {"description": "Absolute difference in selection rates.",
                     "interpretation": "Closer to 0 is better.", "group_rates": group_rates}},
    ]


# ────────────────────────────── Explainability ────────────────────────────────

def _explainability(model, X: pd.DataFrame) -> list[dict]:
    top_features: dict = {}
    mean_imp = 0.0

    if _SHAP:
        try:
            idx = np.random.default_rng(42).choice(len(X), min(200, len(X)), replace=False)
            X_sample = X.iloc[idx]
            explainer = shap.Explainer(model, X_sample)
            sv = explainer(X_sample)
            shap_vals = np.abs(sv.values).mean(axis=0)
            ranked = sorted(zip(X.columns, shap_vals), key=lambda t: t[1], reverse=True)
            top_features = {k: float(v) for k, v in ranked[:10]}
            mean_imp = float(np.mean(shap_vals))
            method = "SHAP"
        except Exception as exc:
            logger.warning("SHAP failed (%s), falling back to feature_importances_", exc)

    if not top_features and hasattr(model, "feature_importances_"):
        fi = model.feature_importances_
        ranked = sorted(zip(X.columns, fi), key=lambda t: t[1], reverse=True)
        top_features = {k: float(v) for k, v in ranked[:10]}
        mean_imp = float(np.mean(fi))
        method = "feature_importances_"
    elif not top_features:
        method = "unavailable"

    complexity = 0
    if hasattr(model, "n_estimators"):
        depth = getattr(model, "max_depth", None) or 0
        complexity = int(model.n_estimators) * max(1, int(2 ** depth) - 1)

    return [
        {"name": "Feature Importance", "category": "explainability", "value": mean_imp,
         "details": {"description": "Mean absolute impact of features on model output.",
                     "method": method, "top_features": top_features}},
        {"name": "Model Complexity", "category": "explainability", "value": float(complexity),
         "details": {"description": "Structural complexity proxy (trees × nodes).",
                     "interpretation": "Lower is more explainable."}},
    ]


# ────────────────────────────── Robustness ────────────────────────────────────

def _robustness(model, X: pd.DataFrame, y: pd.Series, n: int = 20, noise: float = 0.1) -> list[dict]:
    rng = np.random.default_rng(42)
    y_orig = model.predict(X)
    stabilities = []
    for _ in range(n):
        X_noisy = X + rng.normal(0, noise, X.shape)
        y_noisy = model.predict(X_noisy)
        stabilities.append(float(np.mean(y_noisy == y_orig)))

    avg_stability = float(np.mean(stabilities))
    std_stability = float(np.std(stabilities))

    return [
        {"name": "Prediction Stability", "category": "robustness", "value": avg_stability,
         "details": {"description": "Fraction of unchanged predictions under Gaussian noise.",
                     "interpretation": "≥0.90 is acceptable. <0.80 indicates fragility.",
                     "std": std_stability, "noise_level": noise, "n_perturbations": n}},
    ]
