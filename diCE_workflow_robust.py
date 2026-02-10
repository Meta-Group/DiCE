"""
Simple End-to-End DiCE Workflow - Robust Version

Loads a dataset, trains a model, and generates counterfactuals using DiCE.
Uses full dataset access (PublicData mode) for best counterfactual quality.
"""
import pandas as pd
import numpy as np
import argparse
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import dice_ml


def load_and_clean_dataset(sample_size=5000):
    """Load Adult dataset and clean it properly."""
    print("Loading Adult Income dataset...")
    
    # Import from sklearn to check version compatibility
    try:
        from sklearn.datasets import fetch_openml
        dataset = fetch_openml(name='adult', version=1, as_frame=True, parser='auto')
        print(f"  Using sklearn version: {dataset.__version__}")
    except ImportError as e:
        print(f"  ImportError: {e}")
        print("Please check sklearn version compatibility")
        sys.exit(1)
    
    # Fetch adult dataset
    dataset = dataset.frame  # Use .frame property for newer pandas compatibility
    dataset = dataset.dropna(subset=['all'])  # Remove all missing values
    
    # Check what columns we have
    print(f"Available columns: {list(dataset.columns)}")
    
    # Select and rename columns we need
    # Note: Actual Adult dataset uses 'hours-per-week' not 'hours_per_week'
    selected_cols = []
    
    # Continuous features we need
    continuous_cols_to_use = []
    for col in dataset.columns:
        if col == 'age' or col == 'hours-per-week':
            continuous_cols_to_use.append(col)
    
    # Categorical features we need
    categorical_cols_to_use = []
    for col in dataset.columns:
        if col in ['workclass', 'education', 'marital-status', 
                   'occupation', 'race', 'sex']:
            categorical_cols_to_use.append(col)
    
    # Target column
    if 'class' in dataset.columns:
        target_col = 'class'
    elif 'income' in dataset.columns:
        target_col = 'income'
    else:
        print("Warning: Could not find target column, using last available")
        target_col = dataset.columns[-1]
    
    selected_cols = continuous_cols_to_use + categorical_cols_to_use + [target_col]
    
    print(f"Selected features: {selected_cols}")
    
    # Create clean dataframe with only needed columns
    df_clean = dataset[selected_cols].copy()
    
    # Sample if requested
    if sample_size and sample_size < len(df_clean):
        df_clean = df_clean.sample(n=sample_size, random_state=42)
        print(f"  Sampled to {sample_size} rows")
    else:
        print(f"  Using all {len(df_clean)} rows")
    
    return df_clean, continuous_cols_to_use, target_col


def train_model(df_clean, continuous_features, target_col, categorical_features):
    """Train a RandomForest classifier."""
    print("Training machine learning model...")
    
    # Prepare features
    X = df_clean[continuous_features + categorical_features]
    y = df_clean[target_col]
    
    # Encode categorical features
    X_encoded = X.copy()
    fitted_encoders = {}
    
    for col in categorical_features:
        le = LabelEncoder()
        X_encoded[col] = le.fit_transform(X[col].astype(str))
        fitted_encoders[col] = le
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=1,
        max_depth=10
    )
    
    model.fit(X_encoded, y)
    
    train_accuracy = model.score(X_encoded, y) * 100
    print(f"  ✓ Model trained (training accuracy: {train_accuracy:.1f}%)")
    
    return model, fitted_encoders


def train_classifier_and_wrap_for_dice(model, fitted_encoders, feature_names, categorical_features):
    """Wrap trained model for use with DiCE."""
    
    class ModelInterface:
        """Model interface compatible with DiCE."""
        
        def __init__(self, model, feature_names, categorical_features, fitted_encoders):
            self.model = model
            self.feature_names = feature_names
            self.categorical_features = categorical_features
            self.fitted_encoders = fitted_encoders
            self.backend = 'classifier'
            self.model_type = 'classifier'
        
        def get_output(self, input_instance, model_score=True):
            """Get model predictions."""
            if isinstance(input_instance, pd.DataFrame):
                X_input = input_instance.copy()
            elif isinstance(input_instance, dict):
                X_input = pd.DataFrame([input_instance])
            else:
                raise ValueError("input must be DataFrame or dict")
            
            # Encode categorical features using fitted encoders
            X_encoded = X_input.copy()
            for col in self.categorical_features:
                if col in X_input.columns:
                    if hasattr(self, 'fitted_encoders') and col in self.fitted_encoders:
                        encoder = self.fitted_encoders[col]
                        X_encoded[col] = encoder.transform(X_input[col].astype(str))
                    else:
                        # Fallback: handle unknown categories
                        X_encoded[col] = X_input[col].astype(str)
            
            # Get continuous features
            continuous_features = [f for f in self.feature_names if f not in self.categorical_features]
            if continuous_features:
                X_cont = X_input[continuous_features].values
            else:
                X_cont = np.array([[] for _ in range(len(X_input))]).astype(float)
            
            # Combine features
            import numpy as np
            X_final = np.hstack([X_encoded.values, X_cont])
            
            if model_score:
                probs = self.model.predict_proba(X_final)
                return probs
            else:
                preds = self.model.predict(X_final)
                return preds[:1] if len(preds.shape) > 1 else preds
    
    return ModelInterface(model, list(model.feature_names), list(fitted_encoders.keys()))


def main():
    """Main workflow function."""
    parser = argparse.ArgumentParser(
        description='End-to-end DiCE counterfactual explanation workflow (Robust Version)',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--sample-size',
        type=int,
        default=5000,
        help='Number of samples to use (default: 5000)'
    )
    
    parser.add_argument(
        '--method',
        type=str,
        default='genetic',
        choices=['genetic', 'random', 'gradient', 'kdtree'],
        help='DiCE method (default: genetic)'
    )
    
    parser.add_argument(
        '--total-cfs',
        type=int,
        default=4,
        help='Number of counterfactuals to generate (default: 4)'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print="Step 1: Loading Dataset"
    print("=" * 70)
    
    # Load and clean dataset
    df_clean, continuous_features, target_col = load_and_clean_dataset(
        sample_size=args.sample_size
    )
    
    # Train model
    model, encoders = train_model(
        df_clean, 
        continuous_features,
        target_col,
        categorical_features=[f for f in df_clean.columns if f not in continuous_features + [target_col]]
    )
    
    print()
    print("=" * 70)
    print="Step 2: Training ML Model"
    print("=" * 70)
    
    print()
    print("=" * 70)
    print("Step 3: Creating DiCE Explainer")
    print("=" * 70)
    
    # Create DiCE data interface using full dataset (PublicData mode)
    # This gives DiCE access to full training data, enabling all features
    d = dice_ml.Data(
        dataframe=df_clean,  # Use full training dataset
        continuous_features=continuous_features,
        outcome_name=target_col
    )
    
    print(f"  Data interface type: {type(d).__name__}")
    print(f"  Features: {d.feature_names}")
    print(f"  Continuous: {d.continuous_feature_names}")
    print(f"  Categorical: {d.categorical_feature_names}")
    
    # Create model interface with encoding support
    m_interface = train_classifier_and_wrap_for_dice(
        model=model,
        fitted_encoders=encoders,
        feature_names=list(df_clean.columns),
        categorical_features=[f for f in df_clean.columns if f not in continuous_features + [target_col]]
    )
    
    print("  ✓ Explainer created (with full dataset access)")
    
    print()
    print("=" * 70)
    print("Step 4: Selecting Query Instance")
    print("=" * 70)
    
    # Use first sample as query instance
    query_idx = 0
    query_instance = df_clean.iloc[[query_idx]].to_dict()
    
    # Remove target column from query
    if target_col in query_instance:
        del query_instance[target_col]
    
    query_series = pd.DataFrame([query_instance])
    print(f"  Query instance (index {query_idx}):")
    print(query_series.to_string())
    
    # Step 5: Generating Counterfactual Explanations
    print()
    print("=" * 70)
    print="Step 5: Generating Counterfactual Explanations"
    print("=" * 70)
    
    # Create DiCE explainer
    exp = dice_ml.Dice(d, m_interface, method=args.method)
    
    # Generate counterfactuals
    dice_exp = exp.generate_counterfactuals(
        query_instance,
        total_CFs=args.total_cfs,
        desired_class='opposite',
        verbose=True
    )
    
    print()
    print("=" * 70)
    print("Step 6: Displaying Results")
    print("=" * 70)
    
    # Display counterfactual examples
    dice_exp.visualize_as_dataframe(show_only_changes=True)
    
    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print()
    print(f"Dataset used:")
    print(f"  - Total samples: {len(df_clean)}")
    print(f"  - Training samples: {len(df_clean)}")
    print(f"  - Features used: {list(df_clean.columns)}")
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
