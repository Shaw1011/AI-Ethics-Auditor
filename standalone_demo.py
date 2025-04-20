"""
Standalone Demo for AI Ethics Auditor

This demo doesn't require any external dependencies beyond Python's standard library.
"""

import json
import random
import os
import sys
from datetime import datetime

def generate_demo_data(n_samples=100, save_path='demo_data.json'):
    """
    Generate a simple demo dataset without using NumPy
    """
    # Only create directories if there's a directory part in the path
    if os.path.dirname(save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Generate data
    data = []
    for i in range(n_samples):
        # Generate random features
        age = random.randint(18, 80)
        income = random.randint(20000, 150000)
        education = random.randint(8, 22)  # Years of education
        
        # Determine protected attribute (e.g., gender: 0=female, 1=male)
        protected = random.randint(0, 1)
        
        # Introduce bias: higher income people more likely to get loan
        # With additional bias based on protected attribute
        income_factor = (income - 20000) / 130000  # Normalize to 0-1
        protected_bias = 0.2 if protected == 1 else 0
        
        # Calculate loan approval probability with bias
        approval_prob = 0.3 + (0.5 * income_factor) + protected_bias
        
        # Generate outcome
        approved = 1 if random.random() < approval_prob else 0
        
        # Create record
        record = {
            'id': i + 1,
            'age': age,
            'income': income,
            'education': education,
            'protected_attribute': protected,
            'loan_approved': approved
        }
        
        data.append(record)
    
    # Save data to JSON file
    with open(save_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Generated {n_samples} demo records in {save_path}")
    return data

def train_demo_model(name="demo_model", save_path="demo_model.json"):
    """
    Simulate training a model without actually using ML libraries
    """
    # Create a fake model (just a dictionary with rules)
    model = {
        'name': name,
        'type': 'decision_rules',
        'rules': [
            {'feature': 'income', 'threshold': 50000, 'weight': 0.5},
            {'feature': 'age', 'threshold': 25, 'weight': 0.2},
            {'feature': 'education', 'threshold': 16, 'weight': 0.3}
        ],
        'created_at': datetime.now().isoformat()
    }
    
    # Only create directories if there's a directory part in the path
    if os.path.dirname(save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as f:
        json.dump(model, f, indent=2)
    
    print(f"Created demo model at {save_path}")
    return model

def predict_with_demo_model(model, data):
    """
    Make predictions using the demo model
    """
    predictions = []
    
    for record in data:
        score = 0
        for rule in model['rules']:
            feature_val = record.get(rule['feature'], 0)
            if feature_val >= rule['threshold']:
                score += rule['weight']
        
        # Decision threshold
        prediction = 1 if score >= 0.5 else 0
        predictions.append(prediction)
    
    return predictions

def evaluate_demo_model(model, data):
    """
    Evaluate the demo model on provided data
    """
    # Make predictions
    predictions = predict_with_demo_model(model, data)
    
    # Extract actual values
    actuals = [record['loan_approved'] for record in data]
    
    # Calculate metrics
    correct = sum(1 for p, a in zip(predictions, actuals) if p == a)
    accuracy = correct / len(data)
    
    # Calculate metrics by protected attribute
    group_metrics = {}
    for group in [0, 1]:
        group_indices = [i for i, record in enumerate(data) if record['protected_attribute'] == group]
        if group_indices:
            group_pred = [predictions[i] for i in group_indices]
            group_actual = [actuals[i] for i in group_indices]
            group_correct = sum(1 for p, a in zip(group_pred, group_actual) if p == a)
            group_accuracy = group_correct / len(group_indices)
            group_metrics[str(group)] = {
                'count': len(group_indices),
                'accuracy': group_accuracy,
                'approval_rate': sum(group_pred) / len(group_pred)
            }
    
    return {
        'accuracy': accuracy,
        'group_metrics': group_metrics
    }

def run_demo_audit(model, data):
    """
    Run a simplified audit on the demo model
    """
    # Make predictions
    predictions = predict_with_demo_model(model, data)
    
    # Extract actual values and protected attributes
    actuals = [record['loan_approved'] for record in data]
    protected = [record['protected_attribute'] for record in data]
    
    # Calculate overall approval rate
    overall_approval = sum(predictions) / len(predictions)
    
    # Calculate approval rates by protected group
    group_0_indices = [i for i, p in enumerate(protected) if p == 0]
    group_1_indices = [i for i, p in enumerate(protected) if p == 1]
    
    group_0_approval = sum(predictions[i] for i in group_0_indices) / len(group_0_indices) if group_0_indices else 0
    group_1_approval = sum(predictions[i] for i in group_1_indices) / len(group_1_indices) if group_1_indices else 0
    
    # Calculate statistical parity (difference in approval rates)
    statistical_parity = abs(group_1_approval - group_0_approval)
    
    # Calculate disparate impact (ratio of approval rates)
    disparate_impact = min(group_0_approval, group_1_approval) / max(group_0_approval, group_1_approval) if max(group_0_approval, group_1_approval) > 0 else 1
    
    # Calculate feature importance (simplified)
    feature_importance = {rule['feature']: rule['weight'] for rule in model['rules']}
    
    # Create audit results
    audit_results = [
        {
            'name': 'Statistical Parity',
            'category': 'fairness',
            'value': statistical_parity,
            'details': {
                'description': 'Difference in approval rates between groups',
                'interpretation': 'Closer to 0 is better',
                'group_0_approval': group_0_approval,
                'group_1_approval': group_1_approval
            }
        },
        {
            'name': 'Disparate Impact',
            'category': 'fairness',
            'value': disparate_impact,
            'details': {
                'description': 'Ratio of approval rates between groups',
                'interpretation': 'Closer to 1 is better'
            }
        },
        {
            'name': 'Feature Importance',
            'category': 'explainability',
            'value': max(feature_importance.values()),
            'details': {
                'description': 'Importance of each feature in the model',
                'feature_importance': feature_importance
            }
        },
        {
            'name': 'Model Simplicity',
            'category': 'explainability',
            'value': len(model['rules']),
            'details': {
                'description': 'Number of rules in the model',
                'interpretation': 'Lower is more explainable'
            }
        }
    ]
    
    return audit_results

def save_audit_report(audit_results, model_name, save_path="audit_report.json"):
    """
    Save audit results to a JSON file
    """
    report = {
        'model_name': model_name,
        'audit_date': datetime.now().isoformat(),
        'results': audit_results
    }
    
    # Only create directories if there's a directory part in the path
    if os.path.dirname(save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Audit report saved to {save_path}")

def main():
    """Run the demo"""
    print("AI Ethics Auditor Demo")
    print("=" * 50)
    
    # Create data directory
    os.makedirs("data", exist_ok=True)
    
    # Step 1: Generate demo data
    print("\nStep 1: Generating demo data...")
    data = generate_demo_data(n_samples=500, save_path="data/demo_data.json")
    
    # Step 2: Create a demo model
    print("\nStep 2: Creating demo model...")
    model = train_demo_model(save_path="data/demo_model.json")
    
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
    
    # Save audit report
    save_audit_report(audit_results, model['name'], save_path="data/audit_report.json")
    
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
    print(f"All files have been saved in the 'data' directory.")

if __name__ == "__main__":
    main() 