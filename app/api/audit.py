"""
Audit API Module

This module provides API endpoints for managing audits.
"""

from flask import Blueprint, request, jsonify
from sqlalchemy.exc import SQLAlchemyError
from werkzeug.exceptions import NotFound

from app.models.audit import Audit, AuditMetric
from app.models.model import Model
from app.database import db_session
from app.services.audit_service import run_audit

audit_bp = Blueprint('audit', __name__)

@audit_bp.route('/', methods=['GET'])
def get_audits():
    """Get all audits"""
    audits = Audit.query.all()
    return jsonify([{
        'id': audit.id,
        'name': audit.name,
        'model_id': audit.model_id,
        'status': audit.status,
        'created_at': audit.created_at.isoformat()
    } for audit in audits])

@audit_bp.route('/<int:audit_id>', methods=['GET'])
def get_audit(audit_id):
    """Get a specific audit by ID"""
    audit = Audit.query.filter_by(id=audit_id).first()
    if not audit:
        raise NotFound(f"Audit with ID {audit_id} not found")
    
    # Get metrics for this audit
    metrics = [{
        'id': metric.id,
        'name': metric.name,
        'category': metric.category,
        'value': metric.value
    } for metric in audit.metrics]
    
    return jsonify({
        'id': audit.id,
        'name': audit.name,
        'description': audit.description,
        'model_id': audit.model_id,
        'model_name': audit.model.name if audit.model else None,
        'status': audit.status,
        'created_at': audit.created_at.isoformat(),
        'updated_at': audit.updated_at.isoformat(),
        'metrics': metrics
    })

@audit_bp.route('/', methods=['POST'])
def create_audit():
    """Create a new audit"""
    data = request.get_json()
    
    # Validate request
    if not data or not data.get('name') or not data.get('model_id'):
        return jsonify({'error': 'Name and model_id are required'}), 400
    
    # Check if model exists
    model = Model.query.get(data['model_id'])
    if not model:
        return jsonify({'error': 'Model not found'}), 404
    
    try:
        # Create new audit
        audit = Audit(
            name=data['name'],
            model_id=data['model_id']
        )
        db_session.add(audit)
        db_session.commit()
        return jsonify({'message': 'Audit created successfully'}), 201
    except SQLAlchemyError as e:
        db_session.rollback()
        return jsonify({'error': str(e)}), 500 