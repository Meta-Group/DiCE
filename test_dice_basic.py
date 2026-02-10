"""
Simple DiCE test to verify functionality
"""
import dice_ml

# Create a simple PrivateData instance
d = dice_ml.Data(
    features={
        'age': [17, 90],
        'income': [25000, 100000]
    },
    outcome_name='approved'
)

print("DiCE test:")
print(f"  Type: {type(d).__name__}")
print(f"  Features: {d.feature_names}")
print(f"  Continuous: {d.continuous_feature_names}")
print(f"  Categorical: {d.categorical_feature_names}")
