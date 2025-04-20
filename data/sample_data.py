"""
Sample Dataset Generator

This module generates synthetic datasets for testing AI ethics auditing.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

def generate_biased_dataset(n_samples=1000, bias_strength=0.8, random_state=42):
    """
    Generate a synthetic dataset with intentional bias for testing fairness metrics.
    
    Parameters:
    - n_samples: Number of samples to generate
    - bias_strength: Strength of the bias (0 to 1)
    - random_state: Random seed for reproducibility
    
    Returns:
    - X_train, X_test, y_train, y_test: Training and testing data
    - protected_attribute: Array indicating the protected attribute (e.g., gender)
    """
    # Generate a base classification dataset
    X, y = make_classification(
        n_samples=n_samples, 
        n_features=20,
        n_informative=10,
        n_redundant=5,
        random_state=random_state
    )
    
    # Generate a protected attribute (e.g., binary gender for simplicity)
    protected_attribute = np.random.RandomState(random_state).binomial(1, 0.5, size=n_samples)
    
    # Introduce bias: make positive outcomes more likely for protected_attribute=1
    bias_mask = protected_attribute == 1
    n_to_flip = int(np.sum(~bias_mask & (y == 0)) * bias_strength)
    
    # Identify indices to flip
    zero_indices = np.where(~bias_mask & (y == 0))[0]
    if len(zero_indices) > 0:
        flip_indices = np.random.RandomState(random_state).choice(
            zero_indices, size=min(n_to_flip, len(zero_indices)), replace=False
        )
        y[flip_indices] = 1
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test, pa_train, pa_test = train_test_split(
        X, y, protected_attribute, test_size=0.3, random_state=random_state
    )
    
    # Create feature names
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    
    # Convert to DataFrames
    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    X_test_df = pd.DataFrame(X_test, columns=feature_names)
    
    # Add protected attribute to dataframes
    X_train_df['protected_attribute'] = pa_train
    X_test_df['protected_attribute'] = pa_test
    
    return X_train_df, X_test_df, y_train, y_test, pa_train, pa_test

def save_datasets():
    """Generate and save datasets to CSV files"""
    X_train, X_test, y_train, y_test, pa_train, pa_test = generate_biased_dataset()
    
    # Save training data
    train_df = X_train.copy()
    train_df['target'] = y_train
    train_df.to_csv('data/training_data.csv', index=False)
    
    # Save testing data
    test_df = X_test.copy()
    test_df['target'] = y_test
    test_df.to_csv('data/testing_data.csv', index=False)
    
    # Create metadata about the dataset
    metadata = {
        'n_samples_train': len(train_df),
        'n_samples_test': len(test_df),
        'n_features': len(train_df.columns) - 2,  # Excluding target and protected attribute
        'protected_attribute_name': 'protected_attribute',
        'target_name': 'target',
        'protected_group_ratio': np.mean(pa_train),
        'positive_outcome_ratio': np.mean(y_train)
    }
    
    return metadata

if __name__ == "__main__":
    import os
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Generate and save datasets
    metadata = save_datasets()
    print("Datasets generated successfully:")
    print(f"Training samples: {metadata['n_samples_train']}")
    print(f"Testing samples: {metadata['n_samples_test']}")
    print(f"Features: {metadata['n_features']}")
    print(f"Protected group ratio: {metadata['protected_group_ratio']:.2f}")
    print(f"Positive outcome ratio: {metadata['positive_outcome_ratio']:.2f}") 