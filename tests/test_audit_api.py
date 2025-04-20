"""
Audit API Tests

This module tests the Audit API endpoints.
"""

import json

def test_get_audits(client, db):
    """Test getting all audits"""
    response = client.get('/api/audits/')
    assert response.status_code == 200
    assert isinstance(json.loads(response.data), list)

def test_create_audit(client, model):
    """Test creating a new audit"""
    audit_data = {
        'name': 'Test Audit',
        'description': 'Testing audit creation',
        'model_id': model.id
    }
    
    response = client.post(
        '/api/audits/',
        data=json.dumps(audit_data),
        content_type='application/json'
    )
    
    assert response.status_code == 201
    data = json.loads(response.data)
    assert data['name'] == 'Test Audit'
    assert data['status'] == 'pending' 