"""
Audit Service Module

This module provides services for running AI ethics audits.
"""

from datetime import datetime
import os
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker
# Try to import shap, but provide fallback if not available
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# Try to import fairlearn, but provide fallback if not available
try:
    from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference
    from fairlearn.reductions import DemographicParity
    FAIRLEARN_AVAILABLE = True
except ImportError:
    FAIRLEARN_AVAILABLE = False

from app.models.audit import Audit, AuditMetric
from app.models.model import Model
from app.database import db_session as app_db_session, Base
from app.services.model_service import load_model

def run_audit(audit_id):
    """Run an AI ethics audit"""
    # Ensure we have a valid database session
    engine = create_engine('sqlite:///data/app.db')
    db_session = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))
    
    # Get the audit
    audit = db_session.query(Audit).get(audit_id)
    if not audit:
        return
    
    try:
        # Update status to running
        audit.status = 'running'
        db_session.commit()
        
        # Load the model
        model_obj, model_record = load_model(audit.model_id)
        if not model_obj:
            audit.status = 'failed'
            db_session.commit()
            return
        
        # Determine which test data to use
        domain = model_record.model_metadata.get('domain', 'general') if model_record.model_metadata else 'general'
        domain_data_path = f'data/{domain}_data.csv'
        
        # Check for domain-specific test data
        if os.path.exists(domain_data_path):
            test_data_path = domain_data_path
            print(f"Using domain-specific test data for {domain} domain")
        else:
            # Fall back to generic test data
            test_data_path = 'data/testing_data.csv'
            print(f"Domain data not found. Using generic test data")
        
        if not os.path.exists(test_data_path):
            print(f"Test data not found at {test_data_path}")
            audit.status = 'failed'
            db_session.commit()
            return
            
        data = pd.read_csv(test_data_path)
        X = data.drop(['target', 'protected_attribute'], axis=1)
        y = data['target']
        protected_attribute = data['protected_attribute']
        
        # Run fairness metrics
        fairness_metrics = compute_fairness_metrics(model_obj, X, y, protected_attribute)
        
        # Run explainability metrics
        explainability_metrics = compute_explainability_metrics(model_obj, X)
        
        # Run robustness metrics
        robustness_metrics = compute_robustness_metrics(model_obj, X, y)
        
        # Combine all metrics
        all_metrics = []
        all_metrics.extend(fairness_metrics)
        all_metrics.extend(explainability_metrics)
        all_metrics.extend(robustness_metrics)
        
        # Save metrics to database
        for metric_data in all_metrics:
            metric = AuditMetric(
                audit_id=audit.id,
                name=metric_data['name'],
                category=metric_data['category'],
                value=metric_data['value'],
                details=metric_data['details']
            )
            db_session.add(metric)
        
        # Update audit status
        audit.status = 'completed'
        audit.updated_at = datetime.utcnow()
        db_session.commit()
        
    except Exception as e:
        # Handle any errors
        print(f"Error running audit: {str(e)}")
        audit.status = 'failed'
        db_session.commit()
        raise e
    finally:
        # Clean up session
        db_session.remove()

def compute_fairness_metrics(model, X, y, protected_attribute):
    """Compute fairness metrics for the model"""
    # Make predictions
    y_pred = model.predict(X)
    
    if not FAIRLEARN_AVAILABLE:
        # Fallback implementation without fairlearn
        # Calculate simple fairness metrics manually
        
        # Group predictions by protected attribute
        group_0_mask = protected_attribute == 0
        group_1_mask = protected_attribute == 1
        
        # Calculate selection rates for each group
        group_0_selection_rate = y_pred[group_0_mask].mean() if any(group_0_mask) else 0
        group_1_selection_rate = y_pred[group_1_mask].mean() if any(group_1_mask) else 0
        
        # Calculate statistical parity (difference in selection rates)
        statistical_parity = abs(group_0_selection_rate - group_1_selection_rate)
        
        # Calculate disparate impact (ratio of selection rates)
        if max(group_0_selection_rate, group_1_selection_rate) > 0:
            disparate_impact = min(group_0_selection_rate, group_1_selection_rate) / max(group_0_selection_rate, group_1_selection_rate)
        else:
            disparate_impact = 1.0
            
        # Return simplified metrics
        return [
            {
                'name': 'Statistical Parity',
                'category': 'fairness',
                'value': float(statistical_parity),
                'details': {
                    'description': 'Absolute difference in selection rates',
                    'interpretation': 'Closer to 0 is better',
                    'group_0_selection_rate': float(group_0_selection_rate),
                    'group_1_selection_rate': float(group_1_selection_rate),
                    'note': 'Fairlearn not available, using manual calculation'
                }
            },
            {
                'name': 'Disparate Impact Ratio',
                'category': 'fairness',
                'value': float(disparate_impact),
                'details': {
                    'description': 'Ratio of selection rates between groups',
                    'interpretation': 'Closer to 1 is better',
                    'note': 'Fairlearn not available, using manual calculation'
                }
            }
        ]
    
    # Original implementation with fairlearn
    # Calculate demographic parity
    dp_diff = demographic_parity_difference(
        y_true=y,
        y_pred=y_pred,
        sensitive_features=protected_attribute
    )
    
    # Calculate equalized odds
    eo_diff = equalized_odds_difference(
        y_true=y,
        y_pred=y_pred,
        sensitive_features=protected_attribute
    )
    
    # Calculate disparate impact
    y_pred_prob = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else y_pred
    
    # Group acceptance rates
    group_0_acceptance = y_pred_prob[protected_attribute == 0].mean()
    group_1_acceptance = y_pred_prob[protected_attribute == 1].mean()
    
    # Calculate disparate impact ratio (min/max to ensure < 1)
    disparate_impact = min(group_0_acceptance, group_1_acceptance) / max(group_0_acceptance, group_1_acceptance)
    
    # Calculate statistical parity
    statistical_parity = abs(group_0_acceptance - group_1_acceptance)
    
    # Return metrics
    return [
        {
            'name': 'Demographic Parity Difference',
            'category': 'fairness',
            'value': float(dp_diff),
            'details': {
                'description': 'Difference in selection rates between groups',
                'interpretation': 'Closer to 0 is better',
                'group_0_selection_rate': float(y_pred[protected_attribute == 0].mean()),
                'group_1_selection_rate': float(y_pred[protected_attribute == 1].mean())
            }
        },
        {
            'name': 'Equalized Odds Difference',
            'category': 'fairness',
            'value': float(eo_diff),
            'details': {
                'description': 'Max difference in TPR and FPR between groups',
                'interpretation': 'Closer to 0 is better'
            }
        },
        {
            'name': 'Disparate Impact Ratio',
            'category': 'fairness',
            'value': float(disparate_impact),
            'details': {
                'description': 'Ratio of selection rates between groups',
                'interpretation': 'Closer to 1 is better',
                'group_0_acceptance_rate': float(group_0_acceptance),
                'group_1_acceptance_rate': float(group_1_acceptance)
            }
        },
        {
            'name': 'Statistical Parity',
            'category': 'fairness',
            'value': float(statistical_parity),
            'details': {
                'description': 'Absolute difference in selection rates',
                'interpretation': 'Closer to 0 is better'
            }
        }
    ]

def compute_explainability_metrics(model, X):
    """Compute explainability metrics for the model"""
    if not SHAP_AVAILABLE:
        # Fallback implementation if SHAP is not available
        # Calculate basic feature importance if available
        if hasattr(model, 'feature_importances_'):
            # For tree-based models
            feature_importance = dict(zip(X.columns, model.feature_importances_))
            sorted_features = sorted(
                feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )
            top_features = {name: float(importance) for name, importance in sorted_features[:5]}
            mean_importance = float(np.mean(model.feature_importances_))
        else:
            # Default values if no feature importance available
            top_features = {f"feature_{i}": 0.0 for i in range(min(5, len(X.columns)))}
            mean_importance = 0.0
        
        # Calculate model complexity
        if hasattr(model, 'n_estimators'):
            n_trees = model.n_estimators
            max_depth = model.max_depth if model.max_depth is not None else 0
            complexity = n_trees * (2**(max_depth) - 1) if max_depth > 0 else n_trees
        else:
            complexity = 0
        
        return [
            {
                'name': 'Feature Importance',
                'category': 'explainability',
                'value': mean_importance,
                'details': {
                    'description': 'Average impact of features on model output',
                    'note': 'SHAP not available, using built-in feature importance',
                    'top_features': top_features
                }
            },
            {
                'name': 'Model Complexity',
                'category': 'explainability',
                'value': float(complexity),
                'details': {
                    'description': 'Measure of model complexity',
                    'interpretation': 'Lower is generally more explainable'
                }
            }
        ]
    
    # Original implementation with SHAP
    # Sample data for SHAP (for speed)
    sample_indices = np.random.choice(len(X), min(100, len(X)), replace=False)
    X_sample = X.iloc[sample_indices]
    
    # Calculate SHAP values
    explainer = shap.Explainer(model, X_sample)
    shap_values = explainer(X_sample)
    
    # Get mean absolute SHAP values for each feature
    mean_shap_values = np.abs(shap_values.values).mean(axis=0)
    
    # Calculate feature importance from SHAP values
    feature_importance = dict(zip(X.columns, mean_shap_values))
    
    # Sort features by importance
    sorted_features = sorted(
        feature_importance.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    # Get top 5 features
    top_features = {name: float(importance) for name, importance in sorted_features[:5]}
    
    # Calculate feature interaction (simplified)
    feature_interaction = 0.0  # Placeholder
    
    # Calculate model complexity
    if hasattr(model, 'n_estimators'):
        n_trees = model.n_estimators
        max_depth = model.max_depth if model.max_depth is not None else 0
        complexity = n_trees * (2**(max_depth) - 1) if max_depth > 0 else n_trees
    else:
        complexity = 0
    
    # Return metrics
    return [
        {
            'name': 'SHAP Feature Importance',
            'category': 'explainability',
            'value': float(mean_shap_values.mean()),
            'details': {
                'description': 'Average impact of features on model output',
                'top_features': top_features
            }
        },
        {
            'name': 'Model Complexity',
            'category': 'explainability',
            'value': float(complexity),
            'details': {
                'description': 'Measure of model complexity',
                'interpretation': 'Lower is generally more explainable'
            }
        }
    ]

def compute_robustness_metrics(model, X, y, n_perturbations=10, noise_level=0.1):
    """Compute robustness metrics for the model"""
    # Make predictions on original data
    y_pred_orig = model.predict(X)
    
    try:
        # Add Gaussian noise to features
        noise_results = []
        for _ in range(n_perturbations):
            # Add noise to features
            X_noisy = X.copy()
            noise = np.random.normal(0, noise_level, X.shape)
            X_noisy = X_noisy + noise
            
            # Make predictions on noisy data
            y_pred_noisy = model.predict(X_noisy)
            
            # Calculate prediction stability (% of predictions that didn't change)
            stability = np.mean(y_pred_noisy == y_pred_orig)
            noise_results.append(stability)
        
        # Calculate average stability
        average_stability = np.mean(noise_results)
        
        # Return metrics
        return [
            {
                'name': 'Prediction Stability',
                'category': 'robustness',
                'value': float(average_stability),
                'details': {
                    'description': 'Average stability of predictions across multiple perturbations',
                    'interpretation': 'Closer to 1 is better'
                }
            }
        ]
    except Exception as e:
        # Simplified fallback
        return [
            {
                'name': 'Robustness',
                'category': 'robustness',
                'value': 1.0,  # Default value
                'details': {
                    'description': 'Model robustness score',
                    'interpretation': 'Could not compute proper robustness metrics',
                    'error': str(e)
                }
            }
        ] 