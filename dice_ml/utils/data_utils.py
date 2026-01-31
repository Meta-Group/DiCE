"""
Utility functions for extracting metadata from datasets to use with PrivateData interface.
This allows users to pass a full dataset but have DiCE use PrivateData mode
by automatically extracting all necessary metadata (ranges, types, precision, MAD, etc.).
"""
import numpy as np
import pandas as pd
from collections import defaultdict, OrderedDict


def extract_metadata_from_dataset(dataframe, continuous_features, outcome_name, 
                                 compute_mad=True, compute_precision=True):
    """Extracts all metadata needed for PrivateData from a pandas DataFrame.
    
    This function automatically extracts:
    - Feature ranges (min/max for continuous, categories for categorical)
    - Data types (int/float for continuous features)
    - Decimal precision for continuous features
    - Median Absolute Deviation (MAD) for continuous features
    
    :param dataframe: pandas DataFrame containing the training data
    :param continuous_features: list of continuous feature names
    :param outcome_name: name of the target/outcome variable
    :param compute_mad: if True, compute MAD values for continuous features (default: True)
    :param compute_precision: if True, compute decimal precision for continuous features (default: True)
    
    :return: dictionary with all needed parameters for dice_ml.Data(features=...)
    """
    from collections import OrderedDict
    
    # Validate input
    if not isinstance(dataframe, pd.DataFrame):
        raise ValueError("dataframe must be a pandas DataFrame")
    if outcome_name not in dataframe.columns:
        raise ValueError(f"outcome_name '{outcome_name}' not found in dataframe columns")
    
    # Remove outcome column from features
    feature_columns = [col for col in dataframe.columns if col != outcome_name]
    
    # Build features dictionary
    features = OrderedDict()
    features_dict = features  # For OrderedDict support

    # Extract metadata for each feature
    for feature in feature_columns:
        if feature in continuous_features:
            # Continuous feature
            feature_data = dataframe[feature].dropna()

            # Range
            min_val = feature_data.min()
            max_val = feature_data.max()
            features[feature] = [min_val, max_val]

            # Data type (int or float)
            if str(min_val).isdigit() and '.' not in str(min_val) and str(max_val).isdigit():
                features_dict[feature] = 'int'
            else:
                features_dict[feature] = ['float', determine_precision(feature_data)]

        else:
            # Categorical feature - extract unique categories
            feature_values = dataframe[feature].dropna().unique().tolist()
            # Convert to strings to ensure consistency
            features[feature] = [str(val) for val in feature_values]

    # Build PrivateData parameters
    params = {
        'features': features,
        'outcome_name': outcome_name,
    }
    
    # Add MAD if requested
    if compute_mad:
        mad_dict = {}
        for feature in continuous_features:
            if feature in dataframe.columns:
                feature_data = dataframe[feature].dropna().values
                mad_val = np.median(np.abs(feature_data - np.median(feature_data)))
                mad_dict[feature] = mad_val
        params['mad'] = mad_dict
    
    return params


def determine_precision(feature_data, max_modes_to_check=10):
    """Determine the decimal precision of a continuous feature by analyzing the modes.
    
    Uses the maximum precision among the most frequent values in the data.
    
    :param feature_data: pandas Series containing the feature values
    :param max_modes_to_check: maximum number of modes to check
    :return: precision value (0 for int, positive int for float precision)
    """
    modes = feature_data.mode()
    
    if len(modes) == 0:
        return 0  # Fallback to integer
    
    # Use the last mode if there are multiple modes
    modes = modes.head(max_modes_to_check)
    
    max_precision = 0
    for mode_val in modes:
        mode_str = str(mode_val)
        if '.' in mode_str:
            # Float - count decimal places
            decimal_places = len(mode_str.split('.')[-1]) if '.' in mode_str else 0
            if '.' in mode_str:
                decimal_places = len(mode_str.split('.')[-1])
            max_precision = max(max_precision, decimal_places)
        else:
            # Integer - no decimal precision
            pass
    
    return max_precision


def create_private_data_from_dataframe(dataframe, continuous_features, outcome_name):
    """Convenient wrapper that extracts metadata and creates a dice_ml.Data instance.
    
    :param dataframe: pandas DataFrame containing the training data
    :param continuous_features: list of continuous feature names
    :param outcome_name: name of the target/outcome variable
    :return: dice_ml.Data instance using PrivateData interface
    """
    import dice_ml
    
    # Extract all metadata
    params = extract_metadata_from_dataset(
        dataframe=dataframe,
        continuous_features=continuous_features,
        outcome_name=outcome_name,
        compute_mad=True,
        compute_precision=True
    )
    
    # Create Data instance
    d = dice_ml.Data(**params)
    
    return d


if __name__ == '__main__':
    # Example usage
    import dice_ml.utils.helpers as helpers
    
    # Load sample data
    print("Example: Extract metadata from dataset and use with PrivateData\n")
    
    # Create a sample dataset (or load your own)
    sample_data = pd.DataFrame({
        'age': [25, 30, 35, 40, 45, 50, 55, 60],
        'income': [25000, 35000, 45000, 50000, 60000, 75000],
        'education': ['HS-grad', 'Bachelors', 'Masters', 'HS-grad', 'Bachelors', 'Masters'],
        'occupation': ['Service', 'Professional', 'White-Collar', 'Blue-Collar'],
        'workclass': ['Private', 'Self-Employed', 'Private']
    })
    
    continuous_features = ['age', 'income']
    outcome_name = 'education_level'  # Hypothetical outcome
    
    print("\nOriginal dataset:")
    print(sample_data)
    
    # Extract metadata
    params = extract_metadata_from_dataset(
        dataframe=sample_data,
        continuous_features=continuous_features,
        outcome_name='education_level',  # Different from actual columns - for demo
        compute_mad=True,
        compute_precision=True
    )
    
    print("\nExtracted metadata for PrivateData:")
    for key, value in params.items():
        if key == 'features':
            print(f"\n  {key}:")
            for feat_name, feat_range in value.items():
                if isinstance(feat_range, list):\n                    if isinstance(feat_range[0], (int, float)):
                        print(f"    {feat_name}: {feat_range}")
                    else:
                        print(f"    {feat_name}: {feat_range}")
        else:
            if isinstance(value, dict) and key != 'features':
                print(f"\n  {key}:")
                for k, v in value.items():
                    print(f"    {k}: {v:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # Create dice_ml.Data instance
    try:
        d = dice_ml.Data(**params)
        print("\n✓ Successfully created DiCE Data interface with PrivateData!")
        print("\nYou can now use this for counterfactual generation with:")
        print("  - model = dice_ml.Model(...)")
        print("  - exp = dice_ml.Dice(d, m, method='genetic')")
        print("  - exp.generate_counterfactuals(...)")
        
    except Exception as e:
        print(f"\n✗ Error creating DiCE Data interface: {e}")
