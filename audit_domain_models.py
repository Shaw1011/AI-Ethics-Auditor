"""
Audit Domain Models Script for AI Ethics Auditor

This script runs ethical audits on all models created with the create_domain_models.py script.
It will create audit entries and run the evaluation for each model.
"""

import os
from datetime import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker

# Import our app
from app import create_app
from app.database import Base
import app.models.model
import app.models.audit
from app.models.model import Model
from app.models.audit import Audit
from app.services.audit_service import run_audit

def main():
    """Run ethics audits on all available domain models"""
    print("=" * 50)
    print("Running Ethics Audits on Domain Models")
    print("=" * 50)
    
    # Create the Flask app
    app = create_app()
    
    with app.app_context():
        # Set up database connection
        engine = create_engine('sqlite:///data/app.db')
        # Create all tables if they don't exist
        Base.metadata.create_all(bind=engine)
        db_session = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))
        
        # Get all models from the database
        models = db_session.query(Model).all()
        
        if not models:
            print("No models found in the database.")
            print("Please run create_domain_models.py first.")
            return
        
        print(f"Found {len(models)} models to audit.")
        
        # Run audits on each model
        for model in models:
            # Check if model already has completed audits
            existing_audits = db_session.query(Audit).filter_by(
                model_id=model.id, 
                status='completed'
            ).all()
            
            if existing_audits:
                print(f"\nModel '{model.name}' already has {len(existing_audits)} completed audits.")
                choice = input("Do you want to create a new audit for this model? (y/n): ")
                if choice.lower() != 'y':
                    continue
                    
            # Create a new audit
            print(f"\nCreating audit for model: {model.name}")
            
            domain = model.model_metadata.get('domain', 'general') if model.model_metadata else 'general'
            audit = Audit(
                name=f"Ethics Audit for {model.name}",
                description=f"Comprehensive ethics evaluation of {domain} model",
                model_id=model.id,
                status='pending'
            )
            
            # Add and commit to database
            db_session.add(audit)
            db_session.commit()
            print(f"Created audit with ID {audit.id}")
            
            # Run the audit
            try:
                print(f"Running audit for {model.name}...")
                print(f"This may take a few moments...")
                
                # Ensure the testing data exists for this domain
                data_path = f"data/{domain}_data.csv"
                if not os.path.exists(data_path):
                    print(f"Warning: Domain-specific data not found at {data_path}")
                    print("Using general testing data instead.")
                    # Copy the general testing data to this domain
                    import shutil
                    shutil.copy("data/testing_data.csv", data_path)
                
                # Run the actual audit
                run_audit(audit.id)
                
                print(f"Audit completed successfully!")
            except Exception as e:
                print(f"Error running audit: {e}")
                # Update audit status to failed
                audit.status = 'failed'
                db_session.commit()
        
        # Clean up
        db_session.remove()
        
        print("\nAll audits completed!")
        print("=" * 50)
        print("You can now view the audit results in the AI Ethics Auditor web interface.")
        print("=" * 50)

if __name__ == "__main__":
    main() 