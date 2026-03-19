"""Application factory."""
from __future__ import annotations
import logging
import os
from flask import Flask
from app.config import get_config
from app.database import init_db
from app.extensions import cors, jwt, limiter, talisman
from app.middleware.error_handlers import register_error_handlers

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


def create_app(config_class=None) -> Flask:
    app = Flask(__name__, static_folder="static", template_folder="templates")

    cfg = config_class or get_config()
    app.config.from_object(cfg)

    # -------------------------------------------------------------- extensions
    cors.init_app(
        app,
        resources={r"/api/*": {"origins": app.config["CORS_ORIGINS"]}},
        supports_credentials=True,
    )

    jwt.init_app(app)

    limiter.init_app(app)

    is_production = app.config.get("FLASK_ENV") == "production"
    talisman.init_app(
        app,
        force_https=is_production,
        strict_transport_security=is_production,
        content_security_policy={
            "default-src": "'self'",
            "script-src": ["'self'", "cdn.jsdelivr.net", "cdnjs.cloudflare.com"],
            "style-src": ["'self'", "'unsafe-inline'", "cdn.jsdelivr.net", "cdnjs.cloudflare.com"],
            "font-src": ["'self'", "cdnjs.cloudflare.com"],
            "img-src": ["'self'", "data:"],
        } if is_production else False,
        frame_options="DENY",
        x_content_type_options=True,
        referrer_policy="strict-origin-when-cross-origin",
    )

    # -------------------------------------------------------------- jwt callbacks
    from app.models.token_blocklist import TokenBlocklist
    from app.database import get_session

    @jwt.token_in_blocklist_loader
    def _check_blocklist(_jwt_header, jwt_payload: dict) -> bool:
        jti = jwt_payload.get("jti")
        db = get_session()
        return db.query(TokenBlocklist).filter_by(jti=jti).first() is not None

    @jwt.revoked_token_loader
    def _revoked(_jwt_header, _jwt_payload):
        from flask import jsonify
        return jsonify({"error": {"code": "TOKEN_REVOKED", "message": "Token has been revoked."}}), 401

    @jwt.expired_token_loader
    def _expired(_jwt_header, _jwt_payload):
        from flask import jsonify
        return jsonify({"error": {"code": "TOKEN_EXPIRED", "message": "Token has expired."}}), 401

    @jwt.invalid_token_loader
    def _invalid(reason: str):
        from flask import jsonify
        return jsonify({"error": {"code": "INVALID_TOKEN", "message": reason}}), 422

    @jwt.unauthorized_loader
    def _missing(reason: str):
        from flask import jsonify
        return jsonify({"error": {"code": "MISSING_TOKEN", "message": reason}}), 401

    # -------------------------------------------------------------- db + routes
    init_db(app)

    from app.routes import register_routes
    register_routes(app)

    register_error_handlers(app)

    os.makedirs(str(app.config["UPLOAD_DIR"]), exist_ok=True)

    logger.info("AI Ethics Auditor started [env=%s]", app.config.get("FLASK_ENV"))
    return app
