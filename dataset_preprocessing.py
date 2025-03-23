import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib

# Load dataset
df = pd.read_csv('dataset.csv')

# Data cleaning and preprocessing
# ...existing code...

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df.drop('phishing', axis=1))

# Apply PCA
pca = PCA(n_components=30)  # Adjust the number of components as needed
reduced_data = pca.fit_transform(scaled_data)

# Save the scaler and PCA model
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(pca, 'pca.pkl')

# Save the reduced dataset
reduced_df = pd.DataFrame(reduced_data)
reduced_df['phishing'] = df['phishing']
reduced_df.to_csv('reduced_dataset.csv', index=False)