"""
Train Sample Model Script

This script trains a sample model using the synthetic data.
"""

import os
import sys
import json
from datetime import datetime

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.services.model_service import train_model
from app.models.model import Model
from app.database import db_session, init_db
from app import create_app

def main():
    """Train a sample model and save it to the database"""
    # Create app context
    app = create_app()
    with app.app_context():
        # Initialize database
        init_db(app)
        
        # Check if data directory exists
        if not os.path.exists('data'):
            os.makedirs('data', exist_ok=True)
        
        # Check if the sample data exists
        train_data_path = 'data/training_data.csv'
        if not os.path.exists(train_data_path):
            # Generate dataset
            print("Generating sample dataset...")
            from data.sample_data import save_datasets
            save_datasets()
            print("Dataset generated successfully.")
        
        # Train a model
        print("Training sample model...")
        model_info = train_model(
            data_path=train_data_path,
            model_name="sample_model",
            model_type="classification",
            params={'n_estimators': 100, 'max_depth': 5, 'random_state': 42}
        )
        
        # Save model info to database
        model = Model(
            name="Sample Classification Model",
            description="A random forest classifier trained on synthetic data",
            model_type="classification",
            version="1.0",
            model_metadata={
                'model_path': model_info['path'],
                'accuracy': model_info['metrics']['accuracy'],
                'f1_score': model_info['metrics']['f1'],
                'top_features': sorted(
                    model_info['feature_importances'].items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:5]
            },
            user_id=None  # No user associated in this example
        )
        
        db_session.add(model)
        db_session.commit()
        
        print(f"Model trained successfully with ID: {model.id}")
        print(f"Accuracy: {model_info['metrics']['accuracy']:.4f}")
        print(f"F1 Score: {model_info['metrics']['f1']:.4f}")
        print("Top features:")
        for feature, importance in sorted(
            model_info['feature_importances'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]:
            print(f"  - {feature}: {importance:.4f}")

if __name__ == "__main__":
    main() 