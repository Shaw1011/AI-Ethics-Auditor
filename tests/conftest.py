"""
Test Configuration Module

This module provides fixtures for testing.
"""

import pytest
from app import create_app
from app.config import TestingConfig
from app.database import Base, db_session
from app.models.user import User
from app.models.model import Model

@pytest.fixture
def app():
    """Create and configure a Flask app for testing"""
    app = create_app(TestingConfig)
    
    # Establish application context
    with app.app_context():
        yield app

@pytest.fixture
def client(app):
    """A test client for the app"""
    return app.test_client()

@pytest.fixture
def db(app):
    """Set up and tear down the database for tests"""
    # Create tables
    Base.metadata.create_all(bind=db_session.bind)
    
    yield db_session
    
    # Clean up
    db_session.remove()
    Base.metadata.drop_all(bind=db_session.bind)

@pytest.fixture
def user(db):
    """Create a test user"""
    user = User(
        username='testuser',
        email='test@example.com',
        password_hash='hashed_password',
        is_admin=False
    )
    db.add(user)
    db.commit()
    return user

@pytest.fixture
def model(db, user):
    """Create a test model"""
    model = Model(
        name='Test Model',
        description='A model for testing',
        model_type='classification',
        version='1.0',
        metadata={'algorithm': 'random_forest'},
        user_id=user.id
    )
    db.add(model)
    db.commit()
    return model 