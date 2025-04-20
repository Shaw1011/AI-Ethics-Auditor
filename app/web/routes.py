"""
Web Routes Module

This module defines the frontend routes for the application.
"""

from flask import Blueprint, render_template

web_bp = Blueprint('web', __name__, template_folder='../templates')

@web_bp.route('/')
@web_bp.route('/index.html')
def index():
    """Home page route"""
    return render_template('index.html')

@web_bp.route('/models.html')
def models_page():
    """Models page route"""
    return render_template('models.html')

@web_bp.route('/audits.html')
def audits_page():
    """Audits page route"""
    return render_template('audits.html')

@web_bp.route('/create-audit.html')
def create_audit_page():
    """Create audit page route"""
    return render_template('create-audit.html')

@web_bp.route('/audit-details.html')
def audit_details_page():
    """Audit details page route"""
    return render_template('audit-details.html')

@web_bp.route('/model-details.html')
def model_details_page():
    """Model details page route"""
    return render_template('model-details.html') 