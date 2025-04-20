"""
Routes Module

This module registers all routes for the application.
"""

from flask import render_template, send_from_directory

def register_routes(app):
    """Register all routes with the application"""
    from app.api.audit import audit_bp
    from app.api.models import models_bp
    from app.api.users import users_bp
    from app.web.routes import web_bp
    
    # Register blueprints
    app.register_blueprint(audit_bp, url_prefix='/api/audits')
    app.register_blueprint(models_bp, url_prefix='/api/models')
    app.register_blueprint(users_bp, url_prefix='/api/users')
    app.register_blueprint(web_bp)
    
    # Frontend routes
    @app.route('/')
    @app.route('/index.html')
    def index():
        """Home page route"""
        return render_template('index.html')
    
    @app.route('/models.html')
    def models_page():
        """Models page route"""
        return render_template('models.html')
    
    @app.route('/audits.html')
    def audits_page():
        """Audits page route"""
        return render_template('audits.html')
    
    @app.route('/create-audit.html')
    def create_audit_page():
        """Create audit page route"""
        return render_template('create-audit.html')
    
    @app.route('/audit-details.html')
    def audit_details_page():
        """Audit details page route"""
        return render_template('audit-details.html')
    
    @app.route('/model-details.html')
    def model_details_page():
        """Model details page route"""
        return render_template('model-details.html')
    
    # API documentation
    @app.route('/api')
    def api_docs():
        """API documentation route"""
        return {
            "message": "AI Ethics Auditor API",
            "version": "1.0",
            "endpoints": {
                "models": "/api/models",
                "audits": "/api/audits",
                "users": "/api/users"
            }
        } 