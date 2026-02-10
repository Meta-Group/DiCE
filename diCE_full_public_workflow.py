"""
End-to-End DiCE Workflow - Public Data Mode

This script demonstrates the standard way to use DiCE with full dataset access.
Unlike PrivateData mode (only metadata), PublicData mode gives DiCE access to your 
entire training dataset, enabling:
- Post-hoc sparsity refinement
- KD-tree initialization for genetic method
- Data-driven feature weighting (MAD from training data)
- Quantile-based post-hoc guidance
- Overall HIGHER QUALITY counterfactuals

This is the recommended approach in DiCE's README and all example notebooks.

Usage:
    python diCE_full_workflow.py [--dataset-path PATH] [--train-model] [--sample-size N] [--test-size N]
                                   [--method genetic|random|gradient] [--total-cfs N] 
                                   [--query-index N] [--no-train]

Options:
    --dataset-path: Path to CSV file (default: uses Adult dataset)
    --train-model: Load pre-trained model from file instead of training
    --sample-size: Number of samples to use from dataset (default: 5000, all if not --test-size)
    --test-size: Number of samples for testing (if --train-model not set)
    --method: DiCE method - genetic, random, gradient (default: genetic, kdtree not supported for PublicData)
    --total-cfs: Number of counterfactuals to generate (default: 4)
    --query-index: Which sample to use as query instance (default: 0)
    --no-train: Skip model training, use pre-trained model

Features:
- Full workflow with PublicData mode (RECOMMENDED)
- Works with pre-trained or newly trained models
- Better counterfactual quality than PrivateData
- All DiCE methods available
- Automatic data statistics extraction
"""
import pandas as pd
import numpy as np
import argparse
import os
import joblib
import sys
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import dice_ml
from dice_ml.model_interfaces.base_model import BaseModel
from dice_ml.utils import helpers


def load_dataset(dataset_path):
    """Load dataset from CSV file or use Adult dataset by default."""
    print(f"Loading dataset from: {dataset_path}")
    
    if dataset_path:
        try:
            dataset = pd.read_csv(dataset_path)
            print(f"  ✓ Loaded {len(dataset)} rows, {len(dataset.columns)} columns")
            return dataset
        except Exception as e:
            print(f"  ✗ Error loading dataset: {e}")
            print("Falling back to Adult dataset...")
    else:
        # Use built-in Adult dataset
        print("Using built-in Adult Income dataset...")
        try:
            from sklearn.datasets import fetch_openml
            dataset = fetch_openml(name='adult', version=1, as_frame=True, parser='auto').frame
            dataset = dataset.dropna(subset=['all'])
            
            # Standardize column names
            column_mapping = {
                'fnlwgt': 'workclass',
                'education-num': 'education',
                'marital-status': 'marital_status',
                'hours-per-week': 'hours_per_week',
                'native-country': 'native_country'
            }
            dataset = dataset.rename(columns=column_mapping)
            
            # Clean income - convert to binary
            def clean_income(val):
                if val == '<=50K.' or val == '<=50K.':
                    return 0
                elif val == '>50K.' or val == '>50K.':
                    return 1
                else:
                    return int(val)
            
            if 'income' in dataset.columns:
                dataset['income'] = dataset['income'].apply(clean_income)
            
            print(f"  ✓ Loaded {len(dataset)} rows fromAdult dataset")
            return dataset
            
        except ImportError:
            print("  ✗ sklearn.fetch_openml not available")
            print("Please provide a valid CSV file (--dataset-path)")
            return None


def train_or_load_model(model_path, df_train):
    """Train a new model or load a pre-trained model."""
    if not model_path:
        if '--no-train' not in sys.argv:
            print("Training new RandomForest model on dataset...")
            print(f"Using {len(df_train)} training samples")
            
            # Prepare data
            X = df_train[['age', 'workclass', 'fnlwgt', 'education-num', 
                       'marital-status', 'occupation', 'race', 'sex', 
                       'hours-per-week', 'native-country']]
            
            # Target is income (binary)
            y = df_train['income']
            
            # Encode categorical features
            categorical_cols = ['workclass', 'education-num', 'marital-status', 
                            'occupation', 'race', 'sex', 'native-country']
            
            X_encoded = df_train.copy()
            for col in categorical_cols:
                le = LabelEncoder()
                X_encoded[col] = le.fit_transform(df_train[col].astype(str))
            
            # Train Random Forest
            model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=10,
                min_samples_leaf=5,
                n_jobs=1
            )
            
            model.fit(X_encoded, y)
            
            train_accuracy = model.score(X_encoded, y) * 100
            print(f"  ✓ Model trained (training accuracy: {train_accuracy:.1f}%)")
            print()
            
            # Save model
            model_filename = 'dice_rf_model.pkl'
            joblib.dump(model, model_filename)
            print(f"  ✓ Model saved to: {model_filename}")
            
            return model, X_encoded, categorical_cols
    else:
        # Load pre-trained model
        print(f"Loading pre-trained model from: {model_path}")
        try:
            model = joblib.load(model_path)
            print(f"  ✓ Model loaded from: {model_path}")
            
            # Get training data to fit encoders (needed for categorical features)
            # Note: In practice, you would save the full training dataset with the model
            # For this example, we'll fit encoders on a sample
            
            # Sample for fitting encoders (small and fast)
            sample_size = min(1000, len(df_train))
            sample_df = df_train.sample(n=sample_size, random_state=42)
            
            # Fit encoders
            categorical_cols = ['workclass', 'education-num', 'marital-status', 
                            'occupation', 'race', 'sex', 'native-country']
            
            fitted_encoders = {}
            for col in categorical_cols:
                le = LabelEncoder()
                le.fit(sample_df[col].astype(str))
                fitted_encoders[col] = le
            
            return model, fitted_encoders, categorical_cols
        except Exception as e:
            print(f"  ✗ Error loading model: {e}")
            print(f"Please make sure '{model_path}' is a valid .pkl file")
            return None, None, None


def generate_counterfactuals(d, data_interface, model_interface, fitted_encoders, 
                     categorical_cols, query_instance, method='genetic', total_cfs=4):
    """Generate counterfactual explanations using DiCE explainer."""
    print(f"Generating counterfactuals with {method} method...")
    print(f"Query instance: {query_instance}")

    # Explainer object
    exp = dice_ml.Dice(d, model_interface, method=method)
    
    # Generate counterfactuals
    dice_exp = exp.generate_counterfactuals(
        query_instance,
        total_CFs=total_cfs,
        desired_class='opposite',
        verbose=True
    )
    
    print()
    print("=" * 70)
    print("Counterfactual Explanations Generated!")
    print("=" * 70)
    print()
    
    return dice_exp


def main():
    """Main workflow orchestration."""
    parser = argparse.ArgumentParser(
        description='End-to-End DiCE Counterfactual Explanation Workflow with Public Data Mode',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use Adult dataset with default settings
  python diCE_full_workflow.py

  # Use your own dataset
  python diCE_full_workflow.py --dataset-path my_data.csv

  # Use pre-trained model
  python diCE_full_workflow.py --train-model my_model.pkl

  # Use larger dataset
  python diCE_full_workflow.py --sample-size 10000

  # Test first sample
  python diCE_full_workflow.py --query-index 0

  # Use random method for diversity
  python diCE_full_workflow.py --method random --total-cfs 10

Notes:
  • PublicData mode uses your actual training dataset (data_df internally)
  • This enables: post-hoc sparsity, KD-tree init, data-driven feature weighting
  • Best for production use and model explainability
  • Recommended over PrivateData mode (which only stores metadata)
  • Methods: All DiCE methods available (genetic, random, gradient, kdtree)
  • Sample sizes: Adjust based on your data size

For help and more options, use: --help
"""
)
    
    parser.add_argument(
        '--dataset-path',
        type=str,
        default=None,
        help='Path to CSV file (default: uses Adult dataset)'
    )
    
    parser.add_argument(
        '--train-model',
        type=str,
        default=None,
        help='Path to pre-trained .pkl model file (optional, trains new model if not set)'
    )
    
    parser.add_argument(
        '--sample-size',
        type=int,
        default=5000,
        help='Number of samples to use from dataset (default: 5000)'
    )
    
    parser.add_argument(
        '--test-size',
        type=int,
        default=None,
        help='Number of samples for testing (if not set, uses 15%% of dataset, max 80%%)'
    )
    
    parser.add_argument(
        '--method',
        type=str,
        choices=['genetic', 'random', 'gradient', 'kdtree'],
        default='genetic',
        help='DiCE method (default: genetic, kdtree not recommended for PublicData)'
    )
    
    parser.add_argument(
        '--total-cfs',
        type=int,
        default=4,
        help='Number of counterfactuals to generate (default: 4)'
    )
    
    parser.add_argument(
        '--no-train',
        action='store_true',
        help='Skip model training, use pre-trained model from --train-model'
    )
    
    parser.add_argument(
        '--query-index',
        type=int,
        default=0,
        help='Which sample to use as query instance (default: 0)'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("DiCE Full Workflow - Public Data Mode")
    print("=" * 70)
    print()
    print("Configuration:")
    print("  Dataset:", args.dataset_path if args.dataset_path else "Adult dataset (built-in)")
    print("  Sample size:", args.sample_size)
    if args.test_size:
        print("  Test size:", args.test_size)
    print("  DiCE method:", args.method)
    print("  Total CFs:", args.total_cfs)
    print("  Query index:", args.query_index)
    print("  Train model:", "Load pre-trained" if args.train_model else "Train new")
    print()
    
    # Step 1: Load dataset
    df = load_dataset(args.dataset_path)
    if df is None:
        print("Exiting due to dataset loading error.")
        sys.exit(1)
    
    # Split train/test if --test-size is set
    if args.test_size:
        test_size = args.test_size
        if test_size >= len(df):
            parser.error(f"--test-size must be less than dataset size ({len(df)})")
        
        sample_size = test_size
        
        # Get first test_size samples for query
        df_rest = df.drop(df.index[:test_size].index)
        
        print(f"  Using {test_size} test samples at indices {df_rest.index.tolist()[:10]}")
        
        # Prepare test data (with query instances)
        query_instances = df_rest.iloc[args.query_index:args.query_index+test_size]
        if args.query_index + test_size > len(df_rest):
            parser.error(f"--query-index ({args.query_index}) + --test-size ({test_size}) exceeds available samples ({len(df_rest)})")
        
    else:
        query_instances = pd.DataFrame([df.iloc[args.query_index]] )
        test_size = None
    
    print(f"  Total dataset: {len(df)} rows")
    print(f"  Training data: {max(args.sample_size, test_size or len(df) - 1 if test_size else len(df))} rows")
    print()
    
    # Step 2: Train or Load model
    model, fitted_encoders, categorical_cols = train_or_load_model(
        args.train_model, 
        df.iloc[:args.sample_size]
    )
    if model is None:
        print("Exiting due to model loading error.")
        sys.exit(1)
    
    # Step 3: Create DiCE explainer with Public Data mode
    print()
    print("=" * 70)
    print("Step 3: Creating DiCE Explainer with PublicData Mode")
    print("=" * 70)
    print()
    print("  This uses PublicData mode, which gives DiCE full access to your dataset.")
    print("  Advantages vs PrivateData:")
    print("  ✓ Post-hoc sparsity: ENABLED (can refine CFs for proximity)")
    print("  ✓ Feature weighting: Data-driven (MAD from training data)")
    print("  ✓ KD-tree init: Available (for genetic population)")
    print("  ✓ Quantile-based guidance: Available")
    print("  ✓ Actual training data: Used as counterfactuals (for kdtree method)")
    print()
    
    # Note: We're using the full training set (not a subset)
    if args.test_size:
        print(f"  Generating CFs for test set of {test_size} queries")
    else:
        print(f"  Generating CFs for query at index {args.query_index}")
    
    # Create data interface using training data
    # Using PublicData (no PrivateData) with full dataframe access
    # Note: This requires the full dataset, not just metadata
    d = dice_ml.Data(
        dataframe=df.iloc[:args.sample_size],  # Use training data
        continuous_features=['age', 'hours-per-week'],
        outcome_name='income',
        method=dataset  # Ensure PublicData mode with dataset
    )
    
    print("  ✓ Data interface created")
    print()
    
    # Step 4: Generate counterfactuals
    # For test set, process each query instance
    if args.test_size:
        results_list = []
        for i in range(min(test_size, len(query_instances))):
            print(f"  Generating CFs for test instance {i + 1} of {test_size}...")
            try:
                dice_exp_i = generate_counterfactuals(
                    d=d,
                    model_interface=model,
                    fitted_encoders=fitted_encoders,
                    categorical_cols=categorical_cols,
                    query_instance=query_instances.iloc[[i]].to_dict(),
                    method=args.method,
                    total_cfs=args.total_cfs
                )
                results_list.append(dice_exp_i)
            except Exception as e:
                print(f"  Warning: CF generation failed for test instance {i+1}: {e}")
                results_list.append(dice_exp_i)
            
        print(f"  Processed {len(results_list)} test instances")
    else:
        # Single query instance
        print("  Generating CFs for single query instance...")
        try:
            dice_exp = generate_counterfactuals(
                d=d,
                model_interface=model,
                fitted_encoders=fitted_encoders,
                categorical_cols=categorical_cols,
                query_instance=query_instances.iloc[[0]].to_dict(),
                method=args.method,
                total_cfs=args.total_cfs
            )
            results_list = [dice_exp]
        except Exception as e:
            print(f"  Warning: CF generation failed: {e}")
            results_list = [dice_exp]
    
    # Step 5: Display final results
    print()
    print("=" * 70)
    print("Step 5: Visualizing Results")
    print("=" * 70)
    print()
    
    if args.test_size:
        # For test set, aggregate performance
        successful_cfs = sum(1 for dice_exp in results_list if dice_exp.final_cfs_df is not None and len(dice_exp.final_cfs_df) > 0)
        print(f"  Successfully generated CFs for {successful_cfs} of {len(results_list) * args.total_cfs} attempts")
        
        # Calculate average performance metrics
        train_acc = float(model.score(X_encoded.iloc[:args.sample_size], y.iloc[:args.sample_size]))
        test_acc = 0
        valid_results = [r for r in results_list if r.final_cfs_df is not None]
        
        for i, test_query in enumerate(query_instances):
            if valid_results[i]:
                test_query_df = pd.DataFrame([test_query])
                # Get predictions
                test_encoded = test_query_df.copy()
                for col in categorical_cols:
                    test_encoded[col] = fitted_encoders[col].transform(test_query_df[col].astype(str))
                
                test_cont = test_query_df[['age', 'hours-per-week']].values
                
                import numpy as np
                test_final = np.hstack([test_encoded.values, test_cont])
                
                preds = model.predict(test_final)
                test_acc_i = float(model.score(test_encoded, y.iloc[i:test_size+1])) * 100
                test_acc += test_acc_i
        
        if valid_results > 0:
            avg_test_acc = test_acc / len(valid_results)
        print(f"  Average test accuracy on generated CFs: {avg_test_acc:.1f}%")
    
    print()
    print("=" * 70)
    print("Summary:")
    print("=" * 70)
    print(f"  Dataset: {len(df)} rows")
    print(f"  Model: {type(model).__name__}")
    print(f"  Method: {args.method}")
    print(f"  Total CFs requested: {args.total_cfs}")
    print(f"  Query index: {args.query_index}")
    print()
    print("Quality Benefits (PublicData mode):")
    print("  ✓ Full training dataset access (data_df)")
    print("  ✓ Post-hoc sparsity: ENABLED")
    print("  ✓ Data-driven feature weighting (MAD from training data)")
    print("  ✓ KD-tree initialization (for genetic method)")
    print("  ✓ Quantile-based post-hoc guidance")
    print()
    
    if args.test_size:
        print("=" * 70)
        print("Performance Metrics:")
        print(f"  Training accuracy: {train_acc:.1f}%")
        print(f"  Average test accuracy on CFs: {avg_test_acc:.1f}%")
    print(f"  CF generation success rate: {successful_cfs}/{len(results_list) * args.total_cfs} ({successful_cfs/(len(results_list) * args.total_cfs) * 100:.1f}%)")
    
    # Display first query results as example
    if not args.test_size:
        if results_list[0].final_cfs_df is not None:
            print()
            print("Example Counterfactual (first query instance):")
            results_list[0].visualize_as_dataframe(show_only_changes=True)
    
    print()
    print("=" * 70)
    print("Next Steps:")
    print("=" * 70)
    print("  • To use your own dataset:")
    print("    python diCE_full_workflow.py --dataset-path my_data.csv --sample-size <N>")
    print("  • Train separate model and save:")
    print("    python -c \"from sklearn.model_selection import train_test_split; from sklearn.ensemble import RandomForestClassifier; model = RandomForestClassifier(...); model.fit(...)\"")
    print("    python diCE_full_workflow.py --train-model my_model.pkl")
    print("  • To reproduce results consistently:")
    print("    Set random seeds for reproducibility")
    print("  • For best quality: Use larger sample sizes (10000+)")
    print()
    print("Full workflow ready!")
    print("=" * 70)


if __name__ == '__main__':
    main()
