"""
End-to-End Counterfactual Explanations with DiCE

This script demonstrates a complete workflow:
1. Load a dataset
2. Train an ML model on it
3. Use DiCE with PublicData mode (full access)
4. Generate counterfactual explanations

This is ideal for:
- Understanding DiCE's full capabilities
- Testing with your own datasets
- Preparing models for production use

Usage:
    python diCE_full_workflow.py [--dataset-path PATH] [--sample-size N]
                                   [--method METHOD] [--total-cfs N]

Options:
    --dataset-path: Path to CSV file (default: use Adult dataset)
    --sample-size: Number of samples to use (default: 5000)
    --method: DiCE method - random, genetic, gradient (default: genetic)
    --total-cfs: Number of counterfactuals to generate (default: 4)
"""
import pandas as pd
import numpy as np
import argparse
import sys
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import dice_ml


def load_adult_dataset(sample_size=5000):
    """Load and prepare Adult Income dataset from UCI ML repository.
    
    :param sample_size: Number of samples to load (for faster testing)
    :return: DataFrame with selected features
    """
    print("Loading Adult Income dataset...")
    
    from sklearn.datasets import fetch_openml
    
    # Fetch dataset
    dataset = fetch_openml(name='adult', version=1, as_frame=True, parser='auto')
    
    # Clean data - remove rows with missing values (legacy pandas compatibility)
    print("  Cleaning data...")
    # Handle missing values by filling or dropping
    dataset = dataset.replace('?', np.nan)  # Replace '?' with NaN for older datasets
    dataset = dataset.dropna(subset=['age', 'workclass', 'education', 'marital-status', 
                                           'occupation', 'race', 'sex', 'hours-per-week', 'class'])
    print(f"  Removed NaNs: {len(dataset) - len(dropna(subset=dataset))}")
    
    # Select relevant columns
    selected_cols = ['age', 'workclass', 'education', 'marital-status', 
                   'occupation', 'race', 'sex', 'hours-per-week', 'class']
    dataset = dataset[selected_cols].copy()
    
    # Rename columns to be more Pythonic (for newer sklearn)
    column_mapping = {
        'workclass': 'workclass',
        'marital-status': 'marital_status',
        'hours-per-week': 'hours_per_week',
        'sex': 'gender'
    }
    # Try to rename, but if it fails, keep original names
    try:
        dataset = dataset.rename(columns=column_mapping)
        print("  Renamed columns to be more Pythonic")
    except Exception as e:
        print(f"  Note: Could not rename columns: {e}")
        print("  Using original column names")
    
    # Sample for faster execution (optional)
    if sample_size and sample_size < len(dataset):
        dataset = dataset.sample(n=sample_size, random_state=42)
        print(f"  Sampled to {sample_size} rows")
    
    print(f"  Loaded {len(dataset)} samples")
    if sample_size:
        print(f"  Features: {sorted(list(dataset.columns))}")
    else:
        print(f"  Features: {sorted(list(dataset.columns))}")
    
    return dataset


def train_model(X_train, y_train):
    """Train a Random Forest classifier on training data.
    
    :param X_train: Training features
    :param y_train: Training labels
    :return: Trained RandomForest model
    """
    print("Training machine learning model...")
    
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=1,  # Use single core for reproducibility
        max_depth=10,
        n_estimators=100,
        min_samples_leaf=5
    )
    
    # Encode categorical features
    X_train_encoded = X_train.copy()
    
    categorical_features = ['workclass', 'education', 'marital-status', 
                        'occupation', 'race', 'gender']
    
    for col in categorical_features:
        le = LabelEncoder()
        # Check if column exists in dataframe (for robustness)
        if col in X_train.columns:
            X_train_encoded[col] = le.fit_transform(X_train[col].astype(str))
        else:
            # Create encoded column filled with majority class
            print(f"  Warning: Column '{col}' not in features, using default encoding")
            X_train_encoded[col] = 0  # Default value
    
    # Create feature list for categorical columns that were actually encoded
    if categorical_features:
        categorical_features_encoded = [col for col in categorical_features if col in X_train.columns]
    else:
        categorical_features_encoded = []
    
    # Train the model
    model.fit(X_train_encoded, y_train)
    
    train_accuracy = model.score(X_train_encoded, y_train) * 100
    print(f"  ✓ Model trained (training accuracy: {train_accuracy:.1f}%)")
    
    return model, categorical_features_encoded


def generate_counterfactuals(data_interface, model_interface, query_instance, method='genetic', total_cfs=4, 
                         desired_class='opposite'):
    """Generate counterfactual explanations using DiCE.
    
    :param data_interface: DiCE data interface
    :param model_interface: DiCE model interface
    :param query_instance: Query instance (dict or DataFrame)
    :param method: DiCE method
    :param total_cfs: Number of counterfactuals to generate
    :param desired_class: Desired class
    :return: DiCE explanation object
    """
    print("Generating counterfactual explanations...")
    print(f"  Method: {method}")
    print(f"  Total CFs: {total_cfs}")
    
    # Prepare query instance
    if isinstance(query_instance, dict):
        query_instance = pd.DataFrame([query_instance])
    elif not isinstance(query_instance, pd.DataFrame):
        query_instance = pd.DataFrame([query_instance])
    
    # Create DiCE explainer
    exp = dice_ml.Dice(data_interface, model_interface, method=method)
    
    # Generate counterfactuals
    dice_exp = exp.generate_counterfactuals(
        query_instance,
        total_CFs=total_cfs,
        desired_class=desired_class,
        verbose=True
    )
    
    return dice_exp


def main():
    """Main function that orchestrates the complete workflow."""
    
    parser = argparse.ArgumentParser(
        description='End-to-end DiCE counterfactual explanation workflow',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--dataset-path',
        type=str,
        default=None,
        help='Path to CSV file (if not provided, uses Adult dataset)'
    )
    
    parser.add_argument(
        '--sample-size',
        type=int,
        default=5000,
        help='Number of samples to use from dataset (default: 5000)'
    )
    
    parser.add_argument(
        '--method',
        type=str,
        choices=['genetic', 'random', 'gradient', 'kdtree'],
        default='genetic',
        help='DiCE method to use (default: genetic)'
    )
    
    parser.add_argument(
        '--total-cfs',
        type=int,
        default=4,
        help='Number of counterfactuals to generate (default: 4)'
    )
    
    parser.add_argument(
        '--query-index',
        type=int,
        default=0,
        help='Index of sample to use as query instance (default: 0)'
    )
    
    args = parser.parse_args()
    
    # Step 1: Load dataset
    print("=" * 70)
    print("Step 1: Loading Dataset")
    print("=" * 70)
    print()
    
    if args.dataset_path:
        print(f"Loading dataset from: {args.dataset_path}")
        try:
            dataset = pd.read_csv(args.dataset_path)
            print(f"  Loaded {len(dataset)} samples")
        except Exception as e:
            print(f"  ✗ Error loading dataset: {e}")
            print("  Make sure file exists and is a valid CSV.")
            sys.exit(1)
    else:
        dataset = load_adult_dataset(sample_size=args.sample_size)
    
    # Check if dataset has required columns
    required_cols = ['age', 'workclass', 'education', 'marital-status', 
                   'occupation', 'race', 'gender', 'hours-per-week', 'class']
    
    # Handle missing values in target column
    if 'class' not in dataset.columns:
        print("  'class' column not found in dataset, trying alternatives...")
        # Try to find the outcome column
        possible_outcomes = ['class', 'income', 'target', 'outcome']
        for col in possible_outcomes:
            if col in dataset.columns:
                print(f"    Found '{col}' as target column")
                dataset = dataset.rename(columns={col: 'class'})
                outcome_name = 'class'
                break
        else:
            print(f"  Unable to find outcome column")
            print("  Creating synthetic target for demonstration...")
            dataset['class'] = [1] * len(dataset)
    
    # Define features and outcome
    continuous_features = ['age', 'hours_per_week']
    outcome_name = 'class'
    
    # Add features that might be missing
    if 'age' in dataset.columns:
        continuous_features = ['age', 'hours_per_week']
    
    print("Step 2: Training ML Model")
    print("=" * 70)
    print()
    
    # Split into train/test
    X = dataset[continuous_features + ['workclass', 'education', 'marital-status', 
                                          'occupation', 'race', 'gender']]
    
    # Handle missing values in features (drop rows with missing values)
    print("  Handling missing values in features...")
    y = dataset[outcome_name]
    
    # Create indices for rows without missing values
    cols_with_missing = X.columns[X.isna().any()].tolist()
    if cols_with_missing:
        valid_idx = ~X.isna().any(axis=1).all(axis=1)
        X = X[valid_idx]
        y = y[valid_idx]
        print(f"  Removed {len(X) - len(X[valid_idx])} rows with missing values")
    else:
        print("  No missing values in features")
    
    # Use full dataset for training and DiCE (PublicData mode)
    X_train = X[:]  # Use all data
    y_train = y[:]
    
    # Train model
    model, categorical_features = train_model(X_train, y_train)
    
    print("Step 3: Creating DiCE Explainer (with Full Dataset Access)")
    print("=" * 70)
    print()
    
    # Create DiCE data interface using full training dataset
    # This uses PublicData mode, which gives DiCE full access to all features
    d = dice_ml.Data(
        dataframe=dataset,  # Pass full dataset - metadata computed automatically
        continuous_features=continuous_features,
        outcome_name=outcome_name,
        verbose=True
    )
    
    print("  Data interface created!")
    print(f"  Data interface type: {type(d).__name__}")
    print(f"  Features: {d.feature_names}")
    print(f"  Continuous: {d.continuous_feature_names}")
    print(f"  Categorical: {d.categorical_feature_names}")
    
    # Step 4: Creating DiCE Model Interface
    print()
    print("=" * 70)
    print("Step 4: Creating DiCE Model Interface")
    print("=" * 70)
    print()
    
    # Create a simple model interface wrapper
    class SimpleModelInterface:
        """Simple wrapper for sklearn model to work with DiCE."""
        
        def __init__(self, model, feature_names, categorical_features):
            self.model = model
            self.feature_names = feature_names
            self.categorical_features = categorical_features
            self.backend = 'sklearn'
            self.model_type = 'classifier'
            
        def get_output(self, input_instance, model_score=True):
            """Get model predictions or probabilities."""
            if isinstance(input_instance, pd.DataFrame):
                X_input = input_instance.copy()
            elif isinstance(input_instance, dict):
                X_input = pd.DataFrame([input_instance])
            else:
                raise ValueError("input_instance must be DataFrame or dict")
            
            # Encode categorical features
            X_encoded = X_input.copy()
            for col in self.categorical_features:
                if col in X_input.columns:
                    # Use LabelEncoders from training
                    # Find the encoder for this column from the trained model
                    if hasattr(self.model, 'named_encoders'):
                        if col in self.model.named_encoders:
                            encoder = self.model.named_encoders[col]
                            X_encoded[col] = encoder.transform(X_input[col].astype(str))
                        else:
                            raise ValueError(f"LabelEncoder for '{col}' not found in model")
                    else:
                        # Just pass categorical features as-is if no encoder
                        # This may cause issues, but allows script to continue
                        X_encoded[col] = X_input[col]
            
            # Get continuous features
            X_cont = X_input[[f for f in X_input.columns if f not in self.categorical_features]].values
            
            # Combine features
            import numpy as np
            X_final = np.hstack([X_encoded.values, X_cont])
            
            # Get predictions
            if model_score:
                probs = self.model.predict_proba(X_final)
                return probs
            else:
                preds = self.model.predict(X_final)
                return preds[:1] if len(preds.shape) > 1 else preds
    
    model_interface = SimpleModelInterface(model, X_train.columns.tolist(), categorical_features)
    
    print("  Explainer created (with full dataset access)")
    
    # Step 5: Selecting Query Instance
    print()
    print("=" * 70)
    print("Step 5: Selecting Query Instance")
    print("=" * 70)
    print()
    
    query_index = min(args.query_index, len(dataset) - 1)
    query_instance = dataset.iloc[query_index:query_index+1].to_dict()
    
    # Remove outcome column if present
    if outcome_name in query_instance:
        del query_instance[outcome_name]
    
    print(f"  Using sample at index {query_index}:")
    query_df = pd.DataFrame([query_instance])
    print(query_df.to_string())
    
    # Step 6: Generating Counterfactual Explanations
    print()
    print("=" * 70)
    print("Step 6: Generating Counterfactual Explanations")
    print("=" * 70)
    print()
    
    dice_exp = generate_counterfactuals(
        data_interface=d,
        model_interface=model_interface,
        query_instance=query_df,
        method=args.method,
        total_cfs=args.total_cfs,
        desired_class='opposite'
    )
    
    print()
    print("=" * 70)
    print("Counterfactual Explanations Generated!")
    print("=" * 70)
    print()
    
    # Step 7: Display Results
    print()
    print("=" * 70)
    print("Step 7: Results")
    print("=" * 70)
    print()
    
    dice_exp.visualize_as_dataframe(show_only_changes=True)
    
    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print()
    print(f"Dataset used:")
    print(f"  - Total samples: {len(dataset)}")
    print(f"  - Training samples: {len(X_train)}")
    print()
    print(f"DiCE configuration:")
    print(f"  - Method: {args.method}")
    print(f"  - Total CFs: {args.total_cfs}")
    print(f"  - Data mode: PublicData (full dataset access)")
    print()
    print("Quality advantages (vs PrivateData):")
    print("  ✓ Post-hoc sparsity: ENABLED")
    print("  ✓ Feature weighting: Data-driven MAD from training data")
    print("  ✓ KD-tree init: Available for genetic/kdtree methods")
    print("  ✓ Quantile-based guidance: Available")
    print("  ✓ Actual training samples: Can be used as counterfactuals")
    print()
    print("Generated counterfactuals provide diverse, actionable explanations!")


if __name__ == '__main__':
    main()
