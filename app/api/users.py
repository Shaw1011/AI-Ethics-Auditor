"""Users API."""
from __future__ import annotations
import logging
from flask import Blueprint, jsonify
from flask_jwt_extended import get_jwt_identity, jwt_required
from app.database import get_session
from app.models.user import User
from app.utils.errors import forbidden, not_found
from app.utils.pagination import paginate

logger = logging.getLogger(__name__)
users_bp = Blueprint("users", __name__)


def _require_admin():
    db = get_session()
    user = db.query(User).get(int(get_jwt_identity()))
    if not user or not user.is_admin:
        return None, forbidden("Admin access required.")
    return user, None


@users_bp.route("/", methods=["GET"])
@jwt_required()
def list_users():
    _, err = _require_admin()
    if err:
        return err
    db = get_session()
    return jsonify(paginate(db.query(User).order_by(User.created_at.desc())))


@users_bp.route("/<int:user_id>", methods=["GET"])
@jwt_required()
def get_user(user_id: int):
    caller_id = int(get_jwt_identity())
    db = get_session()
    caller = db.query(User).get(caller_id)
    if not caller or (caller_id != user_id and not caller.is_admin):
        return forbidden()
    u = db.query(User).get(user_id)
    if not u:
        return not_found("User")
    return jsonify(u.to_dict())


@users_bp.route("/<int:user_id>", methods=["DELETE"])
@jwt_required()
def delete_user(user_id: int):
    _, err = _require_admin()
    if err:
        return err
    db = get_session()
    u = db.query(User).get(user_id)
    if not u:
        return not_found("User")
    db.delete(u)
    db.commit()
    return jsonify({"message": f"User {user_id} deleted."}), 200
