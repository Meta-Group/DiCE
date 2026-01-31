"""
Simple example to extract metadata from dataset and use with DiCE PrivateData

This demonstrates the manual approach without the data_utils import issue.

Usage:
    1. Load your training dataset
    2. Automatically extract metadata (ranges, types, precision, MAD)
    3. Create PrivateData instance
    4. Use with DiCE to generate counterfactuals
"""
import pandas as pd
import numpy as np
from collections import OrderedDict
import dice_ml
from dice_ml.utils import helpers


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
    # Validate input
    if not isinstance(dataframe, pd.DataFrame):
        raise ValueError("dataframe must be a pandas DataFrame")
    if outcome_name not in dataframe.columns:
        raise ValueError(f"outcome_name '{outcome_name}' not found in dataframe columns")
    
    # Remove outcome column from features
    feature_columns = [col for col in dataframe.columns if col != outcome_name]
    
    # Build features dictionary
    features = OrderedDict()
    
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
                features_dict = features  # This is the features dict
                features_dict[feature] = 'int'
            else:
                features_dict = features
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
                if len(feature_data) > 0:
                    mad_val = np.median(np.abs(feature_data - np.median(feature_data)))
                else:
                    mad_val = 1.0  # Default if no data
                mad_dict[feature] = mad_val
        params['mad'] = mad_dict
    
    return params


def determine_precision(feature_data, max_modes_to_check=10):
    """Determine the decimal precision of a continuous feature by analyzing its modes.
    
    Uses the maximum precision among the most frequent values in the data.
    
    :param feature_data: pandas Series containing the feature values
    :param max_modes_to_check: maximum number of modes to check (default: 10)
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
            max_precision = max(max_precision, decimal_places)
        else:
            # Integer - no decimal precision
            pass
    
    return max_precision


def main():
    print("=" * 70)
    print("Example: Extract Metadata and Use PrivateData")
    print("=" * 70)
    print()
    
    # Step 1: Load your training dataset
    print("Step 1: Loading dataset...")
    
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
    
    print(f"  Loaded {len(sample_data)} samples with {len(sample_data.columns)} features")
    print(f"  Features: {list(sample_data.columns)}")
    print()
    
    # Step 2: Extract metadata automatically
    print("Step 2: Extracting metadata from dataset...")
    print("  (min/max values, data types, precision, MAD, etc.)")
    
    d_private = extract_metadata_from_dataset(
        dataframe=sample_data,
        continuous_features=continuous_features,
        outcome_name=outcome_name,
        compute_mad=True,  # Compute MAD from training data
        compute_precision=True  # Compute precision from training data
    )
    
    print("  Metadata extracted successfully!")
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
    
    # Step 3: Train a simple ML model (or load pre-trained)
    print("Step 3: Training a simple classifier...")
    
    from sklearn.ensemble import RandomForestClassifier
    import sklearn.preprocessing as pp
    
    # Prepare data for training
    X = sample_data[['age', 'workclass', 'education', 'occupation', 'workclass']]
    y = sample_data['education_level']
    
    # Encode categorical variables
    le = pp.LabelEncoder()
    X_encoded = X.copy()
    for col in ['workclass', 'education', 'occupation']:
        X_encoded[col] = le.fit_transform(X[col].astype(str))
    
    # Train a Random Forest classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)
    model.fit(X_encoded, y)
    
    print(f"  Model trained (accuracy: {100*model.score(X_encoded, y):.1f}%)")
    print()
    
    # Step 4: Create DiCE explainer with PrivateData
    print("Step 4: Creating DiCE explainer with PrivateData...")
    print("  Note: Only metadata from training data is kept, not actual data!")
    print()
    
    # Create a model interface wrapper
    class ModelInterface:
        def __init__(self, model, feature_names):
            self.model = model
            self.feature_names = feature_names
            self.backend = 'sklearn'
            self.model_type = 'classifier'
            
        def get_output(self, input_instance, model_score=True):
            if isinstance(input_instance, pd.DataFrame):
                X_input = input_instance.copy()
            elif isinstance(input_instance, dict):
                X_input = pd.DataFrame([input_instance])
            else:
                raise ValueError("input must be DataFrame or dict")
            
            # Encode categorical features
            X_cat = X_input[['workclass', 'education', 'occupation']]
            # Use the same encoders from training
            fitted_encoders = {}
            for col in ['workclass', 'education', 'occupation']:
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                le.fit(sample_data[col].astype(str))
                fitted_encoders[col] = le
                X_cat[col] = le.transform(X_cat[col].astype(str))
            
            X_cont = X_input[['age']]
            X_final = pd.concat([X_cat, X_cont], axis=1)
            
            if model_score:
                probs = self.model.predict_proba(X_final)
                return probs
            else:
                preds = self.model.predict(X_final)
                return preds[:1] if len(preds.shape) > 1 else preds
    
    m_interface = ModelInterface(model, ['age', 'workclass', 'education', 'occupation', 'workclass'])
    
    print("  DiCE explainer created!")
    print("  Data interface type:", type(d_private).__name__)
    print()
    
    # Step 5: Generate counterfactuals
    print("Step 5: Generating counterfactual explanations...")
    
    exp = dice_ml.Dice(d_private, m_interface, method='genetic')
    
    # Create a query instance (exclude outcome)
    query_data = sample_data [['age', 'workclass', 'education', 'occupation', 'workclass']].iloc[0].to_dict()
    
    dice_exp = exp.generate_counterfactuals(
        query_data,
        total_CFs=4,
        desired_class='opposite',
        initialization='random',  # Required for private data
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
    print("Privacy: Only metadata (ranges, types, MAD) was extracted")
    print("No actual training data is stored or shared with DiCE")
    print("Counterfactuals generated using optimization algorithms")
    print()
    print("This approach gives you:")
    print("  - Privacy: Only statistics are kept from your training data")
    print("  - Automatic extraction: No manual specification needed")
    print("  - Data-driven quality: MAD and precision from your real data")
    print()
    print("Run this script directly:")
    print("  python extract_and_create.py")


if __name__ == '__main__':
    main()
