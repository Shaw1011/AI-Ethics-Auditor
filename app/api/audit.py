"""Audits API: CRUD + run trigger + JSON export."""
from __future__ import annotations
import logging

from flask import Blueprint, jsonify, request
from flask_jwt_extended import get_jwt_identity, jwt_required

from app.database import get_session
from app.models.audit import Audit
from app.models.model import Model
from app.services.audit_service import run_audit
from app.services.report_service import generate_json_report
from app.utils.errors import error_response, not_found, forbidden
from app.utils.pagination import paginate
from app.utils.validators import AuditCreateSchema, load_or_422

logger = logging.getLogger(__name__)
audit_bp = Blueprint("audit", __name__)

_create_schema = AuditCreateSchema()


@audit_bp.route("/", methods=["GET"])
def list_audits():
    db = get_session()
    status = request.args.get("status")
    model_id = request.args.get("model_id", type=int)
    q = db.query(Audit).order_by(Audit.created_at.desc())
    if status:
        q = q.filter(Audit.status == status)
    if model_id:
        q = q.filter(Audit.model_id == model_id)
    return jsonify(paginate(q))


@audit_bp.route("/<int:audit_id>", methods=["GET"])
def get_audit(audit_id: int):
    db = get_session()
    a = db.query(Audit).get(audit_id)
    if not a:
        return not_found("Audit")
    return jsonify(a.to_dict())


@audit_bp.route("/", methods=["POST"])
@jwt_required()
def create_audit():
    data, err = load_or_422(_create_schema, request.get_json(silent=True) or {})
    if err:
        return err

    db = get_session()
    if not db.query(Model).get(data["model_id"]):
        return not_found("Model")

    a = Audit(
        name=data["name"],
        description=data.get("description", ""),
        model_id=data["model_id"],
        status="pending",
    )
    db.add(a)
    db.commit()
    logger.info("Audit created id=%d model_id=%d", a.id, a.model_id)
    return jsonify(a.to_dict()), 201


@audit_bp.route("/<int:audit_id>/run", methods=["POST"])
@jwt_required()
def trigger_audit(audit_id: int):
    db = get_session()
    a = db.query(Audit).get(audit_id)
    if not a:
        return not_found("Audit")
    if a.status == "running":
        return error_response("Audit is already running.", code="CONFLICT", status=409)

    # Clear previous metrics for a clean re-run
    for m in list(a.metrics):
        db.delete(m)
    a.status = "pending"
    db.commit()

    try:
        run_audit(audit_id)
    except ValueError as exc:
        return error_response(str(exc), code="AUDIT_FAILED", status=422)
    except Exception as exc:
        logger.error("Audit %d failed: %s", audit_id, exc)
        return error_response("Audit execution failed. Check server logs.", code="AUDIT_FAILED", status=500)

    db.refresh(a)
    return jsonify(a.to_dict())


@audit_bp.route("/<int:audit_id>/export", methods=["GET"])
def export_audit(audit_id: int):
    db = get_session()
    a = db.query(Audit).get(audit_id)
    if not a:
        return not_found("Audit")
    if a.status != "completed":
        return error_response("Only completed audits can be exported.", code="NOT_READY", status=409)
    fmt = request.args.get("format", "json")
    if fmt != "json":
        return error_response("Unsupported format. Use ?format=json", code="UNSUPPORTED_FORMAT", status=400)
    return jsonify(generate_json_report(a))


@audit_bp.route("/<int:audit_id>", methods=["DELETE"])
@jwt_required()
def delete_audit(audit_id: int):
    db = get_session()
    a = db.query(Audit).get(audit_id)
    if not a:
        return not_found("Audit")
    db.delete(a)
    db.commit()
    return jsonify({"message": f"Audit {audit_id} deleted."}), 200
