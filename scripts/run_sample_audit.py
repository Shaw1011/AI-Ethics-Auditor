"""
Run Sample Audit Script

This script runs a sample audit on the trained model.
"""

import os
import sys
import json
from datetime import datetime

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.services.audit_service import run_audit
from app.models.audit import Audit
from app.models.model import Model
from app.database import db_session, init_db
from app import create_app

def main():
    """Run a sample audit on the trained model"""
    # Create app context
    app = create_app()
    with app.app_context():
        # Initialize database
        init_db(app)
        
        # Find the model to audit
        model = Model.query.filter_by(name="Sample Classification Model").first()
        if not model:
            print("No model found. Please run train_sample_model.py first.")
            return
        
        # Create an audit
        audit = Audit(
            name="Sample Ethics Audit",
            description="A comprehensive ethics audit of the sample model",
            model_id=model.id,
            status="pending"
        )
        
        db_session.add(audit)
        db_session.commit()
        
        print(f"Created audit with ID: {audit.id}")
        print("Running audit...")
        
        # Run the audit
        run_audit(audit.id)
        
        # Refresh audit from database
        db_session.refresh(audit)
        
        if audit.status == "completed":
            print("Audit completed successfully!")
            
            # Print audit metrics
            print("\nAudit Results:")
            print("-" * 50)
            
            metrics = audit.metrics
            metrics_by_category = {}
            
            for metric in metrics:
                category = metric.category
                if category not in metrics_by_category:
                    metrics_by_category[category] = []
                metrics_by_category[category].append(metric)
            
            for category, category_metrics in metrics_by_category.items():
                print(f"\n{category.upper()} METRICS:")
                print("-" * 30)
                
                for metric in category_metrics:
                    print(f"{metric.name}: {metric.value:.4f}")
                    if metric.details:
                        if 'interpretation' in metric.details:
                            print(f"  • {metric.details['interpretation']}")
        else:
            print(f"Audit failed with status: {audit.status}")

if __name__ == "__main__":
    main() 