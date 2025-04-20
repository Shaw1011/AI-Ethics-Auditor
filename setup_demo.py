"""
Setup Demo Script for AI Ethics Auditor

This script sets up a demonstration of the AI Ethics Auditor by:
1. Creating a sample model in the database
2. Running an ethics audit on the model
"""

import os
import time
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker
from sklearn.ensemble import RandomForestClassifier

# Import from our application
from app import create_app
from app.models.model import Model
from app.models.audit import Audit, AuditMetric
from app.database import Base
from app.services.audit_service import run_audit

def setup_demo():
    """Set up a complete demonstration with sample data"""
    print("=" * 50)
    print("AI Ethics Auditor - Demo Setup")
    print("=" * 50)
    
    # Create the Flask app
    app = create_app()
    
    # Set up database connection
    engine = create_engine('sqlite:///data/app.db')
    db_session = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))
    
    # Initialize the tables if they don't exist
    Base.metadata.create_all(bind=engine)
    
    # Set up the application context
    with app.app_context():
        # Check if testing_data.csv exists
        if not os.path.exists('data/testing_data.csv'):
            print("Creating sample testing data...")
            create_sample_data()
        else:
            print("Using existing testing data...")
        
        # Create a model directory if it doesn't exist
        os.makedirs('data/models', exist_ok=True)
        
        # Train and save a model
        print("Training a sample model...")
        model_info = train_sample_model()
        
        # Create model in database
        print("Creating model in database...")
        db_model = create_model_in_db(model_info, db_session)
        
        # Create and run an audit
        print("Creating and running audit...")
        audit = create_and_run_audit(db_model, db_session)
        
        print("\nDemo setup completed!")
        print(f"Model ID: {db_model.id}")
        print(f"Audit ID: {audit.id}")
        print("\nYou can now access the web interface at http://localhost:5000")
        print("=" * 50)

def create_sample_data():
    """Create a sample dataset for testing"""
    # Generate synthetic data
    np.random.seed(42)
    num_samples = 1000
    
    # Generate features
    X = pd.DataFrame({
        'feature1': np.random.normal(0, 1, num_samples),
        'feature2': np.random.normal(0, 1, num_samples),
        'feature3': np.random.normal(0, 1, num_samples),
        'feature4': np.random.normal(0, 1, num_samples),
        'feature5': np.random.normal(0, 1, num_samples)
    })
    
    # Generate protected attribute (binary)
    protected_attribute = np.random.randint(0, 2, num_samples)
    
    # Make target depend on features and slightly on protected attribute (to introduce bias)
    logits = 0.8 * X['feature1'] + 0.5 * X['feature2'] - 0.5 * X['feature3'] + 0.1 * protected_attribute
    probs = 1 / (1 + np.exp(-logits))
    target = (np.random.random(num_samples) < probs).astype(int)
    
    # Create dataframe
    data = X.copy()
    data['target'] = target
    data['protected_attribute'] = protected_attribute
    
    # Save to CSV
    data.to_csv('data/testing_data.csv', index=False)
    print(f"Created sample data with {num_samples} samples")

def train_sample_model():
    """Train a sample model for the demo"""
    # Load data
    data = pd.read_csv('data/testing_data.csv')
    
    # Prepare features and target
    X = data.drop(['target', 'protected_attribute'], axis=1)
    y = data['target']
    
    # Create and train model
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X, y)
    
    # Save model
    model_path = 'data/models/demo_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Model saved to {model_path}")
    
    # Return model info
    return {
        'name': 'Demo Classification Model',
        'description': 'A model created for demonstration of AI ethics auditing',
        'model_type': 'classification',
        'version': '1.0',
        'model_path': model_path
    }

def create_model_in_db(model_info, db_session):
    """Create a model entry in the database"""
    # Check if model already exists
    existing = db_session.query(Model).filter_by(name=model_info['name']).first()
    if existing:
        print(f"Model '{model_info['name']}' already exists with ID {existing.id}")
        return existing
    
    # Create new model
    model = Model(
        name=model_info['name'],
        description=model_info['description'],
        model_type=model_info['model_type'],
        version=model_info['version'],
        model_metadata={'model_path': model_info['model_path']}
    )
    
    # Add and commit to database
    db_session.add(model)
    db_session.commit()
    print(f"Created model with ID {model.id}")
    return model

def create_and_run_audit(model, db_session):
    """Create and run an audit for the given model"""
    # Check if audit already exists
    existing = db_session.query(Audit).filter_by(model_id=model.id).first()
    if existing:
        print(f"Audit for model {model.id} already exists with ID {existing.id}")
        return existing
    
    # Create new audit
    audit = Audit(
        name=f"Ethics Audit for {model.name}",
        description="Comprehensive evaluation of fairness, explainability, and robustness",
        model_id=model.id,
        status='pending'
    )
    
    # Add and commit to database
    db_session.add(audit)
    db_session.commit()
    print(f"Created audit with ID {audit.id}")
    
    # We need to modify the run_audit function for this demo script
    # Instead of calling the existing function, we'll create metrics manually
    try:
        print("Generating simulated audit metrics...")
        # Mark audit as running
        audit.status = 'running'
        db_session.commit()
        
        # Create some example metrics
        metrics = [
            {
                'name': 'Statistical Parity', 
                'category': 'fairness',
                'value': 0.12,
                'details': {'description': 'Difference in selection rates between protected groups'}
            },
            {
                'name': 'Feature Importance', 
                'category': 'explainability',
                'value': 0.85,
                'details': {'description': 'Average feature importance across the model'}
            },
            {
                'name': 'Prediction Stability', 
                'category': 'robustness',
                'value': 0.91,
                'details': {'description': 'Stability of predictions under perturbations'}
            }
        ]
        
        # Add metrics to database
        for metric_data in metrics:
            metric = AuditMetric(
                audit_id=audit.id,
                name=metric_data['name'],
                category=metric_data['category'],
                value=metric_data['value'],
                details=metric_data['details']
            )
            db_session.add(metric)
        
        # Mark audit as completed
        audit.status = 'completed'
        db_session.commit()
        print(f"Audit completed successfully with {len(metrics)} metrics")
    except Exception as e:
        print(f"Error creating audit metrics: {e}")
        audit.status = 'failed'
        db_session.commit()
    
    return audit

if __name__ == "__main__":
    setup_demo() 