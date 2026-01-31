"""
Simple example: Extract metadata from dataset and use with DiCE PrivateData

This demonstrates automatic metadata extraction for PrivateData mode
"""
import pandas as pd
import numpy as np
import dice_ml


def main():
    print("=" * 70)
    print("Example: Extract Metadata and Use PrivateData")
    print("=" * 70)
    print()
    
    # Fetch adult dataset directly ( Adult has 'hours_per_week')
    from sklearn.datasets import fetch_openml
    
    # Fetch and prepare Adult dataset
    dataset = fetch_openml(name='adult', version=1, as_frame=True, parser='auto').frame
    dataset = dataset.dropna()
    
    # Use all columns for metadata extraction to match what model expects
    all_columns = ['age', 'workclass', 'education', 'occupation', 'hours_per_week', 'class']
    dataset = dataset[all_columns].sample(n=1000, random_state=42)
    
    print("Step 1: Loading dataset...")
    
    print(f"  Loaded {len(dataset)} samples")
    print()
    
    # Step 2: Define features
    # Use column names that exist in Adult dataset
    continuous_features = ['age']
    outcome_name = 'class'
    
    # Step 3: Extract metadata manually for PrivateData
    print("Step 3: Extracting metadata for PrivateData...")

    # Compute feature ranges
    features_dict = {}

    # Continuous features
    for feature in continuous_features:
        min_val = dataset[feature].min()
        max_val = dataset[feature].max()
        features_dict[feature] = [min_val, max_val]

    # Categorical features
    for feature in ['workclass', 'education', 'occupation']:
        unique_vals = dataset[feature].unique().tolist()
        features_dict[feature] = unique_vals
    
    # Categorical features
    for feature in ['workclass', 'education', 'occupation']:
        unique_vals = dataset[feature].unique().tolist()
        features_dict[feature] = unique_vals
    
    # Compute MAD manually
    mad_dict = {}
    for feature in continuous_features:
        feature_data = dataset[feature].values
        if len(feature_data) > 0:
            mad_val = np.median(np.abs(feature_data - np.median(feature_data)))
        else:
            mad_val = 1.0
        mad_dict[feature] = mad_val
    
    print("  Extracted ranges, categories, and MAD!")
    print()
    
    # Step 4: Create PrivateData instance
    print("Step 3: Creating PrivateData interface...")
    
    d = dice_ml.Data(
        features=features_dict,
        outcome_name=outcome_name,
        mad=mad_dict
    )
    
    print("  PrivateData created!")
    print(f"  Type: {type(d).__name__}")
    print(f"  Features: {d.feature_names}")
    print(f"  Continuous: {d.continuous_feature_names}")
    print(f"  Categorical: {d.categorical_feature_names}")
    print()
    
    # Step 5: Display what was extracted
    print("=" * 70)
    print("Extracted Metadata:")
    print("=" * 70)
    print()
    
    for feature in d.feature_names:
        print(f"\n{feature}:")
        if feature in d.continuous_feature_names:
            feat_range = d.permitted_range[feature]
            feat_mad = d.mad.get(feature, 1.0)
            feat_type = d.type_and_precision.get(feature, 'int')
            print(f"  Type:         {feat_type}")
            print(f"  Range:        [{feat_range[0]}, {feat_range[1]}]")
            print(f"  MAD:         {feat_mad:.4f}")
        else:
            feat_levels = d.categorical_levels.get(feature, [])
            print(f"  Type:         categorical")
            print(f"  Levels:       {len(feat_levels)}")
    
    print()
    print("=" * 70)
    print("Done! You can now use this PrivateData instance with DiCE.")
    print("=" * 70)


if __name__ == '__main__':
    main()
