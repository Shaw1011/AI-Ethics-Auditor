"""Models API: CRUD + secure .joblib upload."""
from __future__ import annotations
import logging
from pathlib import Path

from flask import Blueprint, current_app, jsonify, request
from flask_jwt_extended import get_jwt_identity, jwt_required

from app.database import get_session
from app.models.model import Model
from app.utils.errors import forbidden, not_found, server_error, validation_error
from app.utils.pagination import paginate
from app.utils.security import safe_upload_path, sha256_file
from app.utils.validators import ModelCreateSchema, ModelUpdateSchema, load_or_422

logger = logging.getLogger(__name__)
models_bp = Blueprint("models", __name__)

_create_schema = ModelCreateSchema()
_update_schema = ModelUpdateSchema()

_ALLOWED_EXT = frozenset({".joblib"})
_ALLOWED_MIME = frozenset({"application/octet-stream", "application/x-joblib"})


@models_bp.route("/", methods=["GET"])
def list_models():
    db = get_session()
    q = db.query(Model).order_by(Model.created_at.desc())
    return jsonify(paginate(q))


@models_bp.route("/<int:model_id>", methods=["GET"])
def get_model(model_id: int):
    db = get_session()
    m = db.query(Model).get(model_id)
    if not m:
        return not_found("Model")
    return jsonify(m.to_dict())


@models_bp.route("/", methods=["POST"])
@jwt_required()
def create_model():
    data, err = load_or_422(_create_schema, request.get_json(silent=True) or {})
    if err:
        return err

    db = get_session()
    m = Model(
        name=data["name"],
        description=data.get("description", ""),
        model_type=data["model_type"],
        version=data.get("version", "1.0"),
        model_metadata={},
        user_id=int(get_jwt_identity()),
    )
    db.add(m)
    db.commit()
    logger.info("Model registered id=%d by user=%s", m.id, get_jwt_identity())
    return jsonify(m.to_dict()), 201


@models_bp.route("/<int:model_id>", methods=["PATCH"])
@jwt_required()
def update_model(model_id: int):
    db = get_session()
    m = db.query(Model).get(model_id)
    if not m:
        return not_found("Model")
    if m.user_id is not None and m.user_id != int(get_jwt_identity()):
        return forbidden()

    data, err = load_or_422(_update_schema, request.get_json(silent=True) or {})
    if err:
        return err

    for field in ("name", "description", "version"):
        if field in data:
            setattr(m, field, data[field])

    db.commit()
    return jsonify(m.to_dict())


@models_bp.route("/<int:model_id>", methods=["DELETE"])
@jwt_required()
def delete_model(model_id: int):
    db = get_session()
    m = db.query(Model).get(model_id)
    if not m:
        return not_found("Model")
    if m.user_id is not None and m.user_id != int(get_jwt_identity()):
        return forbidden()

    # Remove stored file if present
    file_path = m.model_metadata.get("file_path")
    if file_path:
        try:
            Path(file_path).unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("Could not delete model file %s: %s", file_path, exc)

    db.delete(m)
    db.commit()
    return jsonify({"message": f"Model {model_id} deleted."}), 200


@models_bp.route("/upload", methods=["POST"])
@jwt_required()
def upload_model():
    """
    Upload a serialised model file (.joblib only).
    The SHA-256 hash is computed at upload time and stored.
    It is verified every time the model is loaded — guaranteeing
    the file has not been tampered with after upload.
    """
    if "file" not in request.files:
        return validation_error({"file": ["No file part in request."]})

    f = request.files["file"]
    if not f.filename:
        return validation_error({"file": ["No filename provided."]})

    ext = Path(f.filename).suffix.lower()
    if ext not in _ALLOWED_EXT:
        return validation_error({"file": [f"Only {', '.join(_ALLOWED_EXT)} files are accepted."]})

    upload_dir: Path = current_app.config["UPLOAD_DIR"]
    upload_dir.mkdir(parents=True, exist_ok=True)

    save_path, safe_name = safe_upload_path(f.filename, upload_dir)

    # Validate form fields first (before saving to disk)
    name = (request.form.get("name") or "").strip()
    model_type = (request.form.get("model_type") or "classification").strip()
    version = (request.form.get("version") or "1.0").strip()
    description = (request.form.get("description") or "").strip()

    data, err = load_or_422(_create_schema, {
        "name": name or safe_name, "model_type": model_type,
        "version": version, "description": description,
    })
    if err:
        return err

    try:
        f.save(str(save_path))
    except OSError as exc:
        logger.error("File save failed: %s", exc)
        return server_error()

    file_hash = sha256_file(save_path)

    db = get_session()
    m = Model(
        name=data["name"],
        description=data["description"],
        model_type=data["model_type"],
        version=data["version"],
        model_metadata={
            "file_path": str(save_path),
            "file_name": safe_name,
            "sha256": file_hash,
            "uploaded": True,
        },
        user_id=int(get_jwt_identity()),
    )
    db.add(m)
    db.commit()
    logger.info("Model file uploaded id=%d hash=%s", m.id, file_hash)
    return jsonify(m.to_dict()), 201
