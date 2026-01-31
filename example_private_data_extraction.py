"""
Example: Extract metadata from a dataset and use with PrivateData

This script demonstrates how to:
1. Load your training dataset
2. Automatically extract all necessary metadata (min/max, types, precision, MAD)
3. Use the extracted metadata with dice_ml.Data (PrivateData mode)
4. Get counterfactual explanations while maintaining data privacy

This approach gives you:
- Privacy: Only metadata is stored, not the actual data
- Convenience: Automatic extraction instead of manual specification
- Accuracy: Metadata computed from your actual training data
"""
import pandas as pd
from dice_ml.utils.data_utils import create_private_data_from_dataframe
import dice_ml
from dice_ml.utils import helpers


def main():
    print("=" * 70)
    print("Example: Using PrivateData Mode with Auto-Extracted Metadata")
    print("=" * 70)
    print()
    
    # Step 1: Load your training dataset
    print("Step 1: Loading dataset...")
    
    # Using adult income dataset as example
    import sklearn
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split
    
    # Fetch the adult dataset
    dataset = fetch_openml(name='adult', version=1, as_frame=True, parser='auto')
    dataset = dataset.frame
    
    # Clean the dataset (handle missing values)
    dataset = dataset.dropna()
    
    # Select a subset of columns for this example
    selected_columns = ['age', 'workclass', 'education', 'marital-status', 
                     'occupation', 'race', 'sex', 'hours-per-week', 'class']
    dataset = dataset[selected_columns].copy()
    
    # Rename columns to be more Pythonic
    dataset.columns = ['age', 'workclass', 'education', 'marital_status',
                     'occupation', 'race', 'gender', 'hours_per_week', 'income']
    
    # For this example, use a smaller subset (5000 samples)
    dataset = dataset.sample(n=5000, random_state=42)
    
    print(f"  Loaded {len(dataset)} samples with {len(dataset.columns)} features")
    print(f"  Features: {list(dataset.columns)}")
    print()
    
    # Step 2: Define continuous features and outcome
    continuous_features = ['age', 'hours_per_week']
    outcome_name = 'income'
    
    # Step 3: Extract metadata automatically
    print("Step 2: Extracting metadata from dataset...")
    print("  (min/max values, data types, precision, MAD, etc.)")
    
    d_private = create_private_data_from_dataframe(
        dataframe=dataset,
        continuous_features=continuous_features,
        outcome_name=outcome_name,
        compute_mad=True,  # Compute MAD from training data
        compute_precision=True  # Compute precision from training data
    )
    
    print("  ✓ Metadata extracted successfully!")
    print(f"  Continuous features: {d_private.continuous_feature_names}")
    print(f"  Categorical features: {d_private.categorical_feature_names}")
    print()
    
    # Display extracted metadata
    print("Extracted metadata:")
    print("-" * 50)
    
    for feature in d_private.feature_names:
        print(f"\nFeature: {feature}")
        if feature in d_private.continuous_feature_names:
            # Continuous feature
            feature_range = d_private.permitted_range[feature]
            feature_mad = d_private.mad.get(feature, 1.0)
            feature_type = d_private.type_and_precision.get(feature, 'int')
            
            print(f"  Type:         {feature_type}")
            print(f"  Range:        [{feature_range[0]}, {feature_range[1]}]")
            print(f"  MAD:         {feature_mad:.4f}")
        else:
            # Categorical feature
            feature_levels = d_private.categorical_levels.get(feature, [])
            print(f"  Type:         categorical")
            print(f"  Levels:       {len(feature_levels)} categories")
            if len(feature_levels) <= 10:
                print(f"                 {feature_levels}")
            else:
                print(f"                 (first 10: {feature_levels[:10]})")
    
    print("-" * 50)
    print()
    
    # Step 4: Extract a query instance and prepare the model
    print("Step 3: Preparing query instance and model...")
    
    # Use a sample query instance from the dataset (exclude outcome)
    query_data = dataset[dataset.columns[:-1]].iloc[0].to_dict()
    
    print(f"  Query instance:")
    for key, value in query_data.items():
        print(f"    {key}: {value}")
    print()
    
    # Step 5: Train a simple ML model (or load pre-trained)
    print("Step 4: Training a simple classifier...")
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    
    # Prepare data for training
    X = dataset[['age', 'workclass', 'education', 'marital_status',
                  'occupation', 'race', 'gender', 'hours_per_week']]
    y = dataset['income']
    
    # Encode categorical variables
    le = LabelEncoder()
    for col in ['workclass', 'education', 'marital_status', 
                  'occupation', 'race', 'gender']:
        X[col] = le.fit_transform(X[col].astype(str))
    
    # Train a Random Forest classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X, y)
    
    # Wrap in dice_ml.Model
    # Note: For this example, we need to handle the encoding in the model
    # In practice, you would typically save the trained model and load it
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    
    print("  ✓ Model trained (accuracy on training data: {:.2%})".format(
        100 * model.score(X, y)))
    print()
    
    # Step 6: Create DiCE explainer with PrivateData
    print("Step 5: Creating DiCE explainer with PrivateData...")
    print("  Note: Only metadata from training data is kept, not the actual data!")
    print()
    
    # Create model interface (using sklearn directly)
    # This is a simplified approach - in practice you would save/load the model
    class SimpleModelInterface:
        def __init__(self, model, feature_names):
            self.model = model
            self.feature_names = feature_names
            self.backend = 'sklearn'
            # Simulate prediction method
            from sklearn.preprocessing import OneHotEncoder
            
            # Create a simple encoder for inference
            X_train = dataframe[['age', 'workclass', 'education', 'marital_status',
                                 'occupation', 'race', 'gender', 'hours_per_week']]
            self.categorical_features = ['workclass', 'education', 'marital_status', 
                                       'occupation', 'race', 'gender']
            self.continuous_features = ['age', 'hours_per_week']
            
            # Fit OHE encoder on categorical columns only
            self.ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
            self.ohe.fit(X_train[self.categorical_features])
            
        def get_output(self, input_instance, model_score=True):
            # Handle both dataframe and dict input
            if isinstance(input_instance, pd.DataFrame):
                X_input = input_instance.copy()
            elif isinstance(input_instance, dict):
                X_input = pd.DataFrame([input_instance])
            else:
                raise ValueError("input_instance must be DataFrame or dict")
            
            # One-hot encode categorical features
            X_cat = self.ohe.transform(X_input[self.categorical_features])
            X_cont = X_input[self.continuous_features].values
            
            # Combine
            import numpy as np
            X_encoded = np.hstack([X_cat, X_cont])
            
            # Get predictions
            if model_score:
                probs = self.model.predict_proba(X_encoded)
                return probs
            else:
                preds = self.model.predict(X_encoded)
                return preds[:1] if hasattr(preds, '__len__') else preds
    
    m = SimpleModelInterface(model, d_private.feature_names)
    
    print("  ✓ DiCE explainer created!")
    print("  Data interface type:", type(d_private).__name__)
    print()
    
    # Step 7: Generate counterfactuals
    print("Step 6: Generating counterfactual explanations...")
    
    exp = dice_ml.Dice(d_private, m, method='genetic')
    
    dice_exp = exp.generate_counterfactuals(
        query_data,
        total_CFs=4,
        desired_class='opposite',
        initialization='random',  # Required for private data (KD-tree not available)
        verbose=True
    )
    
    print()
    print("=" * 70)
    print("Counterfactual Explanations Generated!")
    print("=" * 70)
    print()
    
    # Display results
    dice_exp.visualize_as_dataframe(show_only_changes=True)
    
    print()
    print("=" * 70)
    print("Key Points:")
    print("=" * 70)
    print()
    print("✓ Privacy: Only metadata (ranges, types, MAD) was extracted")
    print("✓ No actual training data is stored or shared")
    print("✓ Counterfactuals generated using optimization algorithms")
    print()
    print("Note: Quality trade-offs with PrivateData:")
    print("  - Post-hoc sparsity enhancement: DISABLED")
    print("  - Feature weighting: Equal (not data-driven)")
    print("  - Population initialization: Random (instead of KD-tree)")
    print("  To improve quality, manually provide more accurate MAD values")
    print()
    
    # Reveal what metadata was extracted
    print("Extracted metadata summary:")
    print() 
    print("Continuous feature metadata:")
    for feat in continuous_features:
        if feat in d_private.mad:
            print(f"  {feat}: MAD={d_private.mad[feat]:.4f}, Range={d_private.permitted_range[feat]}")
    
    print("Categorical feature levels:")
    for feat in d_private.categorical_feature_names:
        if feat in d_private.categorical_levels:
            print(f"  {feat}: {len(d_private.categorical_levels[feat])} levels")


if __name__ == '__main__':
    main()
