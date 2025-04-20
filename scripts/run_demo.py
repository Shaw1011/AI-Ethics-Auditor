"""
Run Demo Script

This script demonstrates the AI Ethics Auditor using simplified demo models
that don't rely on NumPy or other ML libraries.
"""

import os
import sys
import json
from datetime import datetime

# Add the project root to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.demo_service import (
    generate_demo_data, 
    train_demo_model, 
    evaluate_demo_model,
    run_demo_audit
)

def main():
    """Run the demo"""
    print("AI Ethics Auditor Demo")
    print("=" * 50)
    
    # Step 1: Generate demo data
    print("\nStep 1: Generating demo data...")
    data = generate_demo_data(n_samples=500)
    
    # Step 2: Create a demo model
    print("\nStep 2: Creating demo model...")
    model = train_demo_model()
    
    # Step 3: Evaluate the model
    print("\nStep 3: Evaluating model...")
    evaluation = evaluate_demo_model(model, data)
    print(f"Overall accuracy: {evaluation['accuracy']:.4f}")
    
    for group, metrics in evaluation['group_metrics'].items():
        print(f"Group {group} metrics:")
        print(f"  - Count: {metrics['count']}")
        print(f"  - Accuracy: {metrics['accuracy']:.4f}")
        print(f"  - Approval rate: {metrics['approval_rate']:.4f}")
    
    # Step 4: Run audit
    print("\nStep 4: Running ethics audit...")
    audit_results = run_demo_audit(model, data)
    
    print("\nAudit Results:")
    print("=" * 50)
    
    for metric in audit_results:
        print(f"\n{metric['name']} ({metric['category']})")
        print(f"Value: {metric['value']:.4f}")
        print("Details:")
        if 'description' in metric['details']:
            print(f"  - {metric['details']['description']}")
        if 'interpretation' in metric['details']:
            print(f"  - {metric['details']['interpretation']}")
            
        if 'group_0_approval' in metric['details'] and 'group_1_approval' in metric['details']:
            print(f"  - Group 0 approval rate: {metric['details']['group_0_approval']:.4f}")
            print(f"  - Group 1 approval rate: {metric['details']['group_1_approval']:.4f}")
            
        if 'feature_importance' in metric['details']:
            print("  - Feature importance:")
            for feature, importance in metric['details']['feature_importance'].items():
                print(f"    • {feature}: {importance:.4f}")
    
    print("\nDemo completed successfully!")

if __name__ == "__main__":
    main() 