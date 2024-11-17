import joblib
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load the dataset
dataset = pd.read_csv("diabetes.csv")
# Separate the features and the target variable
X = dataset.drop(columns='Outcome', axis=1)
Y = dataset['Outcome']
# Standardize the feature data
scaler = StandardScaler()
standardized_data = scaler.fit_transform(X)

# Save the scaler
joblib.dump(scaler, 'scaler.pkl')