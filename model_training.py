import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, classification_report
import numpy as np
import joblib
import onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# Load reduced dataset
df = pd.read_csv('reduced_dataset.csv')

# Split data
X = df.drop('phishing', axis=1)
y = df['phishing']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Evaluate model with default threshold
y_pred = rf_model.predict(X_test)
print("Classification report with default threshold (0.5):")
print(classification_report(y_test, y_pred))

# Find optimal threshold to reduce false positives
y_prob = rf_model.predict_proba(X_test)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, y_prob)

# Create a dataframe for analysis
threshold_df = pd.DataFrame({
    'threshold': np.append(thresholds, 1.0),  # Add 1.0 to match precision/recall length
    'precision': precision,
    'recall': recall
})

print("\nThreshold analysis:")
print(threshold_df.head())

# Find threshold with high precision (reducing false positives)
high_precision_thresholds = threshold_df[threshold_df['precision'] > 0.95]
if not high_precision_thresholds.empty:
    # Select threshold with best recall among high precision options
    optimal_threshold = high_precision_thresholds.loc[high_precision_thresholds['recall'].idxmax()]['threshold']
    print(f"\nOptimal threshold for high precision (reducing false positives): {optimal_threshold:.4f}")
    
    # Apply optimal threshold
    y_pred_optimal = (y_prob >= optimal_threshold).astype(int)
    print("\nClassification Report with optimal threshold:")
    print(classification_report(y_test, y_pred_optimal))
else:
    # Default to 0.5 if no suitable threshold found
    optimal_threshold = 0.5
    print("\nNo threshold with precision > 0.95 found. Using default threshold of 0.5")

# Save the model
joblib.dump(rf_model, 'rf_model.pkl')

# Save the threshold separately
with open("threshold.txt", "w") as f:
    f.write(str(optimal_threshold))

# Convert Random Forest model to ONNX
initial_type = [('float_input', FloatTensorType([None, X_train.shape[1]]))]
rf_onnx_model = convert_sklearn(rf_model, initial_types=initial_type)
with open("rf_model.onnx", "wb") as f:
    f.write(rf_onnx_model.SerializeToString())

print(f"Model training complete. Threshold set to: {optimal_threshold}")