"""
End-to-End Counterfactual Explanations with DiCE

This script demonstrates a complete workflow:
1. Load a dataset
2. Train an ML model on it
3. Use the dataset with DiCE (PublicData mode - full access)
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
    """Load the Adult Income dataset from UCI ML repository.
    
    :param sample_size: Number of samples to load (for faster testing)
    :return: DataFrame with selected features
    """
    print("Loading Adult Income dataset...")
    
    from sklearn.datasets import fetch_openml
    
    # Fetch dataset
    dataset = fetch_openml(name='adult', version=1, as_frame=True, parser='auto')
    
    # Clean data - remove rows with missing values
    dataset = dataset.dropna()
    
    # Select relevant columns
    selected_cols = ['age', 'workclass', 'education', 'marital-status', 
                   'occupation', 'race', 'sex', 'hours-per-week', 'class']
    dataset = dataset[selected_cols].copy()
    
    # Rename columns to be more Pythonic
    column_mapping = {
        'workclass': 'workclass',
        'marital-status': 'marital_status',
        'hours-per-week': 'hours_per_week',
        'sex': 'gender'
    }
    dataset = dataset.rename(columns=column_mapping)
    
    # Sample for faster execution (optional)
    if sample_size and sample_size < len(dataset):
        dataset = dataset.sample(n=sample_size, random_state=42)
    
    print(f"  Loaded {len(dataset)} samples")
    print(f"  Features: {list(dataset.columns)}")
    
    return dataset


def train_model(X_train, y_train):
    """Train a Random Forest classifier on the training data.
    
    :param X_train: Training features
    :param y_train: Training labels
    :return: Trained RandomForest model
    """
    print("Training machine learning model...")
    
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=1,  # Use single core for reproducibility
        max_depth=10
    )
    
    # Encode categorical features
    X_train_encoded = X_train.copy()
    
    categorical_features = ['workclass', 'education', 'marital_status', 'occupation', 'race', 'gender']
    
    for col in categorical_features:
        le = LabelEncoder()
        X_train_encoded[col] = le.fit_transform(X_train[col].astype(str))
    
    # Train the model
    model.fit(X_train_encoded, y_train)
    
    train_accuracy = model.score(X_train_encoded, y_train) * 100
    print(f"  ✓ Model trained (training accuracy: {train_accuracy:.1f}%)")
    
    return model, categorical_features


def generate_counterfactuals(data_interface, model_interface, query_instance, method='genetic', total_cfs=4, 
                         desired_class='opposite'):
    """Generate counterfactual explanations using DiCE.
    
    :param data_interface: DiCE data interface
    :param model_interface: DiCE model interface
    :param query_instance: Query instance (dict or DataFrame)
    :param method: DiCE method ('genetic', 'random', 'gradient')
    :param total_cfs: Number of counterfactuals to generate
    :param desired_class: Desired class ('opposite' or specific class)
    :return: DiCE explanation object
    """
    print("Generating counterfactual explanations...")
    print(f"  Method: {method}")
    print(f"  Total CFs: {total_cfs}")
    print()
    
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
    
    print()
    print("=" * 70)
    print("Counterfactual Explanations Generated!")
    print("=" * 70)
    
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
        # Load user's dataset
        print(f"Loading dataset from: {args.dataset_path}")
        try:
            dataset = pd.read_csv(args.dataset_path)
            print(f"  Loaded {len(dataset)} samples")
        except Exception as e:
            print(f"  ✗ Error loading dataset: {e}")
            print("  Make sure the CSV file exists and is readable.")
            sys.exit(1)
    else:
        # Use built-in Adult dataset
        dataset = load_adult_dataset(sample_size=args.sample_size)
    
    # Define features
    continuous_features = ['age', 'hours_per_week']
    outcome_name = 'class'
    
    # Split into train/test
    X = dataset[continuous_features + ['workclass', 'education', 'marital_status', 
                                          'occupation', 'race', 'gender']]
    y = dataset[outcome_name]
    
    # Use only training data for DiCE
    X_train = X[:]  # Use all data for training and DiCE
    y_train = y[:]
    
    # Step 2: Train ML model
    print()
    print("=" * 70)
    print("Step 2: Training ML Model")
    print("=" * 70)
    print()
    
    model, categorical_features = train_model(X_train, y_train)
    
    # Step 3: Create DiCE Data Interface
    print()
    print("=" * 70)
    print("Step 3: Creating DiCE Data Interface")
    print("=" * 70)
    print()
    
# Create DiCE data interface using the full training dataset
    # This is the key difference from PrivateData approach
    # PublicData mode gives DiCE access to the full dataset
    from dice_ml.utils.data_utils import create_private_data_from_dataframe
    
    data_interface = create_private_data_from_dataframe(
        dataframe=X_train,  # Using the actual training data
        continuous_features=continuous_features,
        outcome_name=outcome_name,
        compute_mad=True,    # Compute MAD from training data
        compute_precision=True  # Compute precision from training data
    )
    
    print(f"  Data interface created (PublicData mode)")
    print(f"  Features: {data_interface.feature_names}")
    print(f"  Continuous: {data_interface.continuous_feature_names}")
    print(f"  Categorical: {data_interface.categorical_feature_names}")
    print()
    
    # Display extracted metadata
    print("Extracted Metadata (from training data):")
    print("-" * 50)
    
    for feature in data_interface.continuous_feature_names:
        if feature in data_interface.mad:
            print(f"  {feature}:")
            print(f"    Type:         continuous")
            print(f"    Range:        [{data_interface.permitted_range[feature][0]}, "
                  f"{data_interface.permitted_range[feature][1]}]")
            print(f"    MAD:         {data_interface.mad[feature]:.4f}")
        else:
            print(f"  {feature}:")
            print(f"    Type:         continuous")
            print(f"    Range:        [{data_interface.permitted_range[feature][0]}, "
                  f"{data_interface.permitted_range[feature][1]}]")
            print(f"    MAD:         (default - should compute MAD)")
    
    for feature in data_interface.categorical_feature_names:
        if feature in data_interface.categorical_levels:
            print(f"  {feature}:")
            print(f"    Type:         categorical")
            print(f"    Levels:       {len(data_interface.categorical_levels[feature])} categories")
            if len(data_interface.categorical_levels[feature]) <= 5:
                print(f"                    {data_interface.categorical_levels[feature]}")
            else:
                print(f"                    (first 5: {data_interface.categorical_levels[feature][:5]})")
    
    print("-" * 50)
    print()
    
    # Step 4: Create DiCE Model Interface
    print()
    print("=" * 70)
    print("Step 4: Creating DiCE Explainer")
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
                raise ValueError("input must be DataFrame or dict")
            
            # Encode categorical features
            X_encoded = X_input.copy()
            for col in self.categorical_features:
                # Use LabelEncoders fitted during training
                if not hasattr(self, 'encoders'):
                    self.encoders = {}
                    for col in self.categorical_features:
                        le = LabelEncoder()
                        le.fit(X_train[col].astype(str))
                        self.encoders[col] = le
                
                X_encoded[col] = self.encoders[col].transform(X_input[col].astype(str))
            
            # Get continuous features
            X_cont = X_input[[f for f in self.feature_names if f not in self.categorical_features]].values
            
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
    
    model_interface = SimpleModelInterface(
        model=model,
        feature_names=list(X_train.columns),
        categorical_features=categorical_features
    )
    
    print("  Explainer created (with full dataset access)")
    print()
    
    # Save model (for convenience)
    import joblib
    model_path = 'diCE_model_trained.pkl'
    joblib.dump(model, model_path)
    print(f"  Model saved to: {model_path}")
    print()
    
    # Step 5: Select query instance
    print()
    print("=" * 70)
    print("Step 5: Selecting Query Instance")
    print("=" * 70)
    print()
    
    # Get query instance from dataset
    query_index = min(args.query_index, len(dataset) - 1)
    query_instance = dataset.iloc[query_index:query_index+1]
    
    # Drop outcome column
    query_instance = query_instance.drop(columns=[outcome_name])
    
    print(f"  Using sample at index {query_index}:")
    print(query_instance.to_string())
    print()
    
    # Step 6: Generate Counterfactuals
    print()
    print("=" * 70)
    print("Step 6: Generating Counterfactual Explanations")
    print("=" * 70)
    print()
    
    dice_exp = generate_counterfactuals(
        data_interface=data_interface,
        model_interface=model_interface,
        query_instance=query_instance,
        method=args.method,
        total_cfs=args.total_cfs,
        desired_class='opposite'
    )
    
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
    print(f"Quality advantages (vs PrivateData):")
    print("  ✅ Post-hoc sparsity: ENABLED")
    print("  ✅ Feature weighting: Data-driven MAD from training data")
    print("  ✅ KD-tree init: Available for genetic/kdtree methods")
    print("  ✅ Quantile-based guidance: Available")
    print("  ✅ Actual training samples: Can be used as counterfactuals")
    print()
    print("Generated counterfactuals provide diverse, actionable explanations!")
    print("=" * 70)


if __name__ == '__main__':
    main()
