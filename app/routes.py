"""Blueprint registration."""
from flask import Flask, jsonify


def register_routes(app: Flask) -> None:
    from app.api.audit import audit_bp
    from app.api.auth import auth_bp
    from app.api.models import models_bp
    from app.api.users import users_bp
    from app.web.routes import web_bp

    app.register_blueprint(auth_bp, url_prefix="/api/auth")
    app.register_blueprint(models_bp, url_prefix="/api/models")
    app.register_blueprint(audit_bp, url_prefix="/api/audits")
    app.register_blueprint(users_bp, url_prefix="/api/users")
    app.register_blueprint(web_bp)

    @app.route("/api")
    def api_root():
        return jsonify({
            "name": "AI Ethics Auditor API",
            "version": "2.0",
            "endpoints": {
                "auth": "/api/auth",
                "models": "/api/models",
                "audits": "/api/audits",
                "users": "/api/users",
            },
        })
