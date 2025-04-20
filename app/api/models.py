"""
Models API Module

This module provides API endpoints for managing AI models.
"""

from flask import Blueprint, request, jsonify
from sqlalchemy.exc import SQLAlchemyError

from app.models.model import Model
from app.database import db_session

models_bp = Blueprint('models', __name__)

@models_bp.route('/', methods=['GET'])
def get_models():
    """Get all models"""
    models = Model.query.all()
    return jsonify([{
        'id': model.id,
        'name': model.name,
        'model_type': model.model_type,
        'version': model.version
    } for model in models])

@models_bp.route('/<int:model_id>', methods=['GET'])
def get_model(model_id):
    """Get a specific model by ID"""
    model = Model.query.get_or_404(model_id)
    
    return jsonify({
        'id': model.id,
        'name': model.name,
        'description': model.description,
        'model_type': model.model_type,
        'version': model.version,
        'metadata': model.model_metadata,
        'user_id': model.user_id,
        'created_at': model.created_at.isoformat(),
        'updated_at': model.updated_at.isoformat()
    })

@models_bp.route('/', methods=['POST'])
def create_model():
    """Create a new model"""
    data = request.get_json()
    
    # Validate request
    if not data or not data.get('name') or not data.get('model_type'):
        return jsonify({'error': 'Name and model_type are required'}), 400
    
    try:
        # Create new model
        model = Model(
            name=data['name'],
            description=data.get('description', ''),
            model_type=data['model_type'],
            version=data.get('version', '1.0'),
            model_metadata=data.get('metadata', {}),
            user_id=data.get('user_id')  # Make user_id optional
        )
        
        db_session.add(model)
        db_session.commit()
        
        return jsonify({
            'id': model.id,
            'name': model.name,
            'model_type': model.model_type,
            'version': model.version
        }), 201
        
    except SQLAlchemyError as e:
        db_session.rollback()
        return jsonify({'error': str(e)}), 500

@models_bp.route('/<int:model_id>', methods=['PUT'])
def update_model(model_id):
    """Update a model"""
    model = Model.query.get_or_404(model_id)
    data = request.get_json()
    
    try:
        if 'name' in data:
            model.name = data['name']
        if 'description' in data:
            model.description = data['description']
        if 'model_type' in data:
            model.model_type = data['model_type']
        if 'version' in data:
            model.version = data['version']
        if 'metadata' in data:
            model.model_metadata = data['metadata']
        
        db_session.commit()
        
        return jsonify({
            'id': model.id,
            'name': model.name,
            'model_type': model.model_type,
            'version': model.version
        })
        
    except SQLAlchemyError as e:
        db_session.rollback()
        return jsonify({'error': str(e)}), 500 