"""Report generation service."""
from __future__ import annotations
from datetime import datetime, timezone
from app.models.audit import Audit


def generate_json_report(audit: Audit) -> dict:
    """Return a structured JSON audit report with a risk summary."""
    metrics_by_cat: dict[str, list] = {}
    for m in audit.metrics:
        metrics_by_cat.setdefault(m.category, []).append(m.to_dict())

    # Risk summary
    summary: dict[str, object] = {"overall_risk": "low", "flags": []}
    flags: list[str] = []

    fairness = metrics_by_cat.get("fairness", [])
    dp = next((m for m in fairness if "Demographic Parity" in m["name"]), None)
    di = next((m for m in fairness if "Disparate Impact" in m["name"]), None)

    if dp and dp["value"] > 0.1:
        flags.append(f"Demographic Parity Difference is {dp['value']:.4f} (>0.10 threshold).")
    if di and di["value"] < 0.8:
        flags.append(f"Disparate Impact Ratio is {di['value']:.4f} (<0.80 legal threshold).")

    robustness = metrics_by_cat.get("robustness", [])
    stab = next((m for m in robustness if "Stability" in m["name"]), None)
    if stab and stab["value"] < 0.85:
        flags.append(f"Prediction Stability is {stab['value']:.4f} (<0.85 threshold).")

    if len(flags) >= 3:
        summary["overall_risk"] = "high"
    elif len(flags) >= 1:
        summary["overall_risk"] = "medium"

    summary["flags"] = flags

    return {
        "schema_version": "2.0",
        "report_type": "AI Ethics Audit Report",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "audit": {
            "id": audit.id,
            "name": audit.name,
            "description": audit.description,
            "status": audit.status,
            "created_at": audit.created_at.isoformat(),
            "completed_at": audit.updated_at.isoformat(),
        },
        "model": {
            "id": audit.model.id,
            "name": audit.model.name,
            "type": audit.model.model_type,
            "version": audit.model.version,
        } if audit.model else None,
        "metrics": metrics_by_cat,
        "summary": summary,
    }
