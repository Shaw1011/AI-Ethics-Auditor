"""
Create Domain Models Script for AI Ethics Auditor

This script creates multiple AI models from different domains for demonstration:
1. Healthcare prediction model
2. Financial risk model
3. Hiring recommendation model
4. Customer segmentation model

Each model is trained on synthetic data with deliberate bias patterns.
"""

import os
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker

# Import from our application
from app import create_app
from app.database import Base
import app.models.model  # This will register the Model class
import app.models.audit  # This will register the Audit class
from app.models.model import Model

def main():
    """Set up domain-specific models for AI Ethics Auditor demo"""
    print("=" * 50)
    print("Creating Domain-Specific Models for AI Ethics Auditor")
    print("=" * 50)
    
    # Create the Flask app and set up the application context
    app = create_app()
    
    with app.app_context():
        # Set up database connection
        engine = create_engine('sqlite:///data/app.db')
        # Create all tables if they don't exist
        Base.metadata.create_all(bind=engine)
        db_session = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))
        
        # Create a model directory if it doesn't exist
        os.makedirs('data/models', exist_ok=True)
        
        # Create domain-specific datasets and models
        domain_configs = [
            {
                "name": "Healthcare Risk Predictor",
                "description": "Predicts patient readmission risk based on health indicators",
                "model_type": "classification",
                "version": "1.0",
                "class_name": "RandomForestClassifier",
                "domain_name": "healthcare",
                "bias_feature": "age",  # Older patients might be classified differently
                "class_args": {"n_estimators": 100, "max_depth": 8, "random_state": 42}
            },
            {
                "name": "Loan Default Predictor",
                "description": "Predicts likelihood of loan default based on financial history",
                "model_type": "classification",
                "version": "1.0",
                "class_name": "LogisticRegression",
                "domain_name": "finance",
                "bias_feature": "income",  # Lower income applicants might be disadvantaged
                "class_args": {"C": 1.0, "random_state": 42}
            },
            {
                "name": "Hiring Recommendation System",
                "description": "Predicts candidate suitability based on resume data",
                "model_type": "classification", 
                "version": "1.0",
                "class_name": "GradientBoostingClassifier",
                "domain_name": "hr",
                "bias_feature": "gender",  # Gender bias in hiring
                "class_args": {"n_estimators": 100, "learning_rate": 0.1, "random_state": 42}
            },
            {
                "name": "Customer Churn Predictor",
                "description": "Predicts which customers are likely to stop using a service",
                "model_type": "classification",
                "version": "1.0", 
                "class_name": "SVC",
                "domain_name": "marketing",
                "bias_feature": "location",  # Geographic bias
                "class_args": {"probability": True, "C": 10.0, "gamma": 'scale', "random_state": 42}
            }
        ]
        
        # Process each domain model
        for config in domain_configs:
            print(f"\nCreating {config['name']}...")
            
            # Create domain dataset
            dataset_path = f"data/{config['domain_name']}_data.csv"
            create_domain_dataset(dataset_path, domain=config['domain_name'], bias_feature=config['bias_feature'])
            
            # Train and save the model
            model_info = train_domain_model(dataset_path, config)
            
            # Register the model in the database
            register_model_in_db(model_info, db_session)
        
        # Clean up the session
        db_session.remove()
        
        print("\nAll domain models created successfully!")
        print("=" * 50)
        print("You can now use these models in the AI Ethics Auditor")
        print("=" * 50)

def create_domain_dataset(filepath, domain="healthcare", bias_feature="age", num_samples=1000):
    """Create a synthetic dataset for a specific domain with designed bias patterns"""
    np.random.seed(42 + hash(domain) % 100)  # Different seed for each domain
    
    # Features common to all datasets
    data = {
        'feature1': np.random.normal(0, 1, num_samples),
        'feature2': np.random.normal(0, 1, num_samples),
        'feature3': np.random.normal(0, 1, num_samples),
        'feature4': np.random.normal(0, 1, num_samples),
        'feature5': np.random.normal(0, 1, num_samples)
    }
    
    # Domain-specific protected attributes
    if domain == "healthcare":
        # Age bias (0: younger, 1: older)
        protected_attribute = np.random.binomial(1, 0.5, num_samples)
        # Feature meanings for healthcare
        data['blood_pressure'] = np.random.normal(120, 20, num_samples)
        data['glucose_level'] = np.random.normal(100, 15, num_samples)
        data['heart_rate'] = np.random.normal(75, 10, num_samples)
        
    elif domain == "finance":
        # Income bias (0: higher income, 1: lower income)
        protected_attribute = np.random.binomial(1, 0.4, num_samples)
        # Feature meanings for finance
        data['credit_score'] = np.random.normal(700, 100, num_samples)
        data['debt_ratio'] = np.random.normal(0.3, 0.2, num_samples)
        data['loan_amount'] = np.random.normal(10000, 5000, num_samples)
        
    elif domain == "hr":
        # Gender bias (0: male, 1: female) - simplified binary for demo
        protected_attribute = np.random.binomial(1, 0.5, num_samples)
        # Feature meanings for HR
        data['years_experience'] = np.random.normal(5, 3, num_samples)
        data['education_level'] = np.random.normal(16, 2, num_samples)  # Years of education
        data['interview_score'] = np.random.normal(7, 2, num_samples)
        
    elif domain == "marketing":
        # Location bias (0: urban, 1: rural)
        protected_attribute = np.random.binomial(1, 0.3, num_samples)
        # Feature meanings for marketing
        data['purchase_frequency'] = np.random.normal(5, 3, num_samples)
        data['customer_lifetime_value'] = np.random.normal(1000, 500, num_samples)
        data['months_active'] = np.random.normal(24, 12, num_samples)
    else:
        # Default case
        protected_attribute = np.random.binomial(1, 0.5, num_samples)
    
    # Create a biased target variable
    # The bias is created by making the protected attribute influence the target
    base_probability = 0.5 + 0.2 * data['feature1'] - 0.1 * data['feature2'] + 0.3 * data['feature3']
    
    # Add bias based on protected attribute (stronger influence than it should have)
    # This creates a model that may potentially discriminate based on the protected attribute
    bias_strength = 0.3  # Strength of the bias
    biased_probability = base_probability + bias_strength * protected_attribute
    biased_probability = np.clip(biased_probability, 0.05, 0.95)  # Ensure valid probabilities
    
    # Generate target
    target = np.random.binomial(1, biased_probability)
    
    # Create dataframe
    df = pd.DataFrame(data)
    df['target'] = target
    df['protected_attribute'] = protected_attribute
    
    # Save to CSV
    df.to_csv(filepath, index=False)
    print(f"Created {domain} dataset with {num_samples} samples at {filepath}")
    print(f"Bias feature: {bias_feature}, proportion of protected class: {protected_attribute.mean():.2f}")
    
    return filepath

def train_domain_model(dataset_path, config):
    """Train a model for a specific domain using the appropriate algorithm"""
    # Load data
    data = pd.read_csv(dataset_path)
    
    # Prepare features and target
    X = data.drop(['target', 'protected_attribute'], axis=1)
    y = data['target']
    
    # Create the appropriate model based on configuration
    if config['class_name'] == 'RandomForestClassifier':
        model = RandomForestClassifier(**config['class_args'])
    elif config['class_name'] == 'LogisticRegression':
        model = LogisticRegression(**config['class_args'])
    elif config['class_name'] == 'GradientBoostingClassifier':
        model = GradientBoostingClassifier(**config['class_args'])
    elif config['class_name'] == 'SVC':
        model = SVC(**config['class_args'])
    else:
        # Default to RandomForest
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Train the model
    model.fit(X, y)
    
    # Save the model
    model_path = f"data/models/{config['domain_name']}_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    # Calculate training accuracy
    train_accuracy = model.score(X, y)
    print(f"Model trained with {model.__class__.__name__}, accuracy: {train_accuracy:.4f}")
    print(f"Model saved to {model_path}")
    
    # Create and return model info
    return {
        'name': config['name'],
        'description': config['description'],
        'model_type': config['model_type'],
        'version': config['version'],
        'model_path': model_path,
        'domain': config['domain_name'],
        'algorithm': config['class_name'],
        'training_accuracy': train_accuracy
    }

def register_model_in_db(model_info, db_session):
    """Register a model in the database"""
    # Check if model already exists
    existing = db_session.query(Model).filter_by(name=model_info['name']).first()
    if existing:
        print(f"Model '{model_info['name']}' already exists with ID {existing.id}")
        return existing
    
    # Create model metadata
    metadata = {
        'model_path': model_info['model_path'],
        'domain': model_info['domain'],
        'algorithm': model_info['algorithm'],
        'training_accuracy': model_info['training_accuracy']
    }
    
    # Create new model
    model = Model(
        name=model_info['name'],
        description=model_info['description'],
        model_type=model_info['model_type'],
        version=model_info['version'],
        model_metadata=metadata,
        # Make user_id optional - it can be null
        user_id=None
    )
    
    # Add and commit to database
    db_session.add(model)
    db_session.commit()
    print(f"Registered model '{model_info['name']}' in database with ID {model.id}")
    return model

if __name__ == "__main__":
    main() 