"""
Ultra-simple PrivateData example - no external dependencies

This demonstrates using PrivateData mode with a minimal, working example
"""
import dice_ml
import pandas as pd


def main():
    print("=" * 70)
    print("Ultra-Simple PrivateData Example")
    print("=" * 70)
    print()
    print("This example shows how to use PrivateData mode")
    print("to provide only metadata (ranges, types, MAD) without storing data.")
    print()
    
    # Step 1: Create PrivateData with manual metadata
    print("Step 1: Creating PrivateData with metadata...")
    
    d = dice_ml.Data(
        features={
            'age': [17, 90],
            'workclass': ['Government', 'Other/Unknown', 'Private', 'Self-Employed'],
            'education': ['Assoc', 'Bachelors', 'Doctorate', 'HS-grad', 
                         'Masters', 'Prof-school', 'School', 'Some-college'],
            'occupation': ['Blue-Collar', 'Other/Unknown', 'Professional', 
                        'Sales', 'Service', 'White-Collar'],
            'hours_per_week': [1, 99]
        },
        outcome_name='income',
        # Optional: manually provide MAD values
        # mad={
        #     'age': 13.35,
        #     'hours_per_week': 12.5
        # }
    )
    
    print("  PrivateData created!")
    print(f"  Type: {type(d).__name__}")
    print(f"  Features: {d.feature_names}")
    print(f"  Continuous: {d.continuous_feature_names}")
    print(f"  Categorical: {d.categorical_feature_names}")
    print()
    
    # Step 2: Display metadata summary
    print("=" * 70)
    print("Metadata Provided:")
    print("=" * 70)
    print()
    
    for feature in d.feature_names:
        print(f"\n  Feature: {feature}")
        if feature in d.continuous_feature_names:
            feat_range = d.permitted_range.get(feature, [None, None])
            feat_mad = d.mad.get(feature, 1.0)
            feat_type = d.type_and_precision.get(feature, 'unknown')
            
            print(f"    Type:         continuous")
            print(f"    Range:        [{feat_range[0]}, {feat_range[1]}]")
            print(f"    MAD (default): {feat_mad:.4f}")
            print()
            print("    TIP: You can improve quality by providing accurate MAD from training data")
            print("          Example: dice_ml.Data(features=..., mad={'age': 13.35, ...})")
        else:
            feat_levels = d.categorical_levels.get(feature, [])
            
            print(f"    Type:         categorical")
            print(f"    Levels:       {len(feat_levels)} categories")
            if len(feat_levels) <= 5:
                for level in feat_levels:
                    print(f"                    - {level}")
            else:
                print(f"                    First 5: {feat_levels[:5]}")
                print(f"                    ... ({len(feat_levels) - 5} more)")
    
    print()
    print("=" * 70)
    print("Next Steps:")
    print("=" * 70)
    print("1. Train your ML model separately (e.g., with sklearn, PyTorch, TensorFlow)")
    print("2. Create dice_ml.Model() with your trained model")
    print("3. Create dice_ml.Dice(d, m, method='genetic')")
    print("4. Generate counterfactuals!")
    print()
    print("Note on methods:")
    print("  - genetic/random/gradient: All work with PrivateData")
    print("  - kdtree: NOT available (requires full data with data_df)")
    print()
    print("For better counterfactual quality with PrivateData:")
    print("  - Provide accurate MAD values from training data via 'mad' parameter")
    print("  - This enables proper feature weighting in proximity loss")
    print("  - Without MAD, all features weighted equally")
    print()
    print("See METADATA_EXTRACTION_README.md for advanced usage with automatic extraction")
    print("=" * 70)


if __name__ == '__main__':
    main()
