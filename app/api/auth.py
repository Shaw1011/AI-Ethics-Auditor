"""Authentication API: register, login, refresh, logout."""
from __future__ import annotations
import logging
from datetime import datetime, timezone

from flask import Blueprint, request, jsonify
from flask_jwt_extended import (
    create_access_token,
    create_refresh_token,
    get_jwt,
    get_jwt_identity,
    jwt_required,
)
from sqlalchemy.exc import IntegrityError

from app.database import get_session
from app.extensions import limiter
from app.models.token_blocklist import TokenBlocklist
from app.models.user import User
from app.utils.errors import conflict, error_response, validation_error
from app.utils.validators import LoginSchema, RegisterSchema, load_or_422

logger = logging.getLogger(__name__)
auth_bp = Blueprint("auth", __name__)

_register_schema = RegisterSchema()
_login_schema = LoginSchema()


@auth_bp.route("/register", methods=["POST"])
@limiter.limit("10 per hour")
def register():
    data, err = load_or_422(_register_schema, request.get_json(silent=True) or {})
    if err:
        return err

    db = get_session()
    if db.query(User).filter_by(username=data["username"]).first():
        return conflict("Username already taken.")
    if db.query(User).filter_by(email=data["email"]).first():
        return conflict("Email already registered.")

    user = User(username=data["username"], email=data["email"], is_admin=data.get("is_admin", False))
    user.set_password(data["password"])

    try:
        db.add(user)
        db.commit()
    except IntegrityError:
        db.rollback()
        return conflict("Username or email already exists.")

    logger.info("New user registered: %s", user.username)
    return jsonify({
        "access_token": create_access_token(identity=str(user.id)),
        "refresh_token": create_refresh_token(identity=str(user.id)),
        "user": user.to_dict(),
    }), 201


@auth_bp.route("/login", methods=["POST"])
@limiter.limit("20 per 5 minutes")
def login():
    data, err = load_or_422(_login_schema, request.get_json(silent=True) or {})
    if err:
        return err

    db = get_session()
    user = db.query(User).filter_by(username=data["username"]).first()

    # Constant-time comparison even on non-existent user (prevent user enumeration)
    if not user or not user.check_password(data["password"]) or not user.is_active:
        return error_response("Invalid credentials.", code="INVALID_CREDENTIALS", status=401)

    logger.info("User logged in: %s", user.username)
    return jsonify({
        "access_token": create_access_token(identity=str(user.id)),
        "refresh_token": create_refresh_token(identity=str(user.id)),
        "user": user.to_dict(),
    })


@auth_bp.route("/refresh", methods=["POST"])
@jwt_required(refresh=True)
def refresh():
    """Issue a new access token using a valid refresh token."""
    identity = get_jwt_identity()
    return jsonify({"access_token": create_access_token(identity=identity)})


@auth_bp.route("/logout", methods=["DELETE"])
@jwt_required(verify_type=False)
def logout():
    """Revoke both access and refresh tokens."""
    payload = get_jwt()
    db = get_session()
    entry = TokenBlocklist(jti=payload["jti"], token_type=payload["type"])
    db.add(entry)
    db.commit()
    logger.info("Token revoked: jti=%s", payload["jti"])
    return jsonify({"message": "Successfully logged out."}), 200


@auth_bp.route("/me", methods=["GET"])
@jwt_required()
def me():
    db = get_session()
    user = db.query(User).get(int(get_jwt_identity()))
    if not user or not user.is_active:
        return error_response("User not found.", code="NOT_FOUND", status=404)
    return jsonify({"user": user.to_dict()})
