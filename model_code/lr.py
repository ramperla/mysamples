import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
import joblib

# Load the dataset
df = pd.read_csv('testdata.csv')

# Ensure column names are stripped of whitespace
df.columns = df.columns.str.strip()

print("Initial dataset shape:", df.shape)

# Print unique values in 'isFraud' column before conversion
print("\nUnique values in 'isFraud' column before conversion:", df['isFraud'].unique())
print("Value counts for 'isFraud' before conversion:")
print(df['isFraud'].value_counts(dropna=False))

# Convert 'WITHDRAWAL_AMT', 'DEPOSIT_AMT', and 'BALANCE_AMT' to numeric
numeric_columns = ['WITHDRAWAL_AMT', 'DEPOSIT_AMT', 'BALANCE_AMT']
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col].str.replace(',', '').str.replace(' ', ''), errors='coerce')

# Fill NaN values with 0 for WITHDRAWAL_AMT and DEPOSIT_AMT
df['WITHDRAWAL_AMT'] = df['WITHDRAWAL_AMT'].fillna(0)
df['DEPOSIT_AMT'] = df['DEPOSIT_AMT'].fillna(0)

# Convert 'isFraud' to numeric (0 for FALSE, 1 for TRUE)
df['isFraud'] = df['isFraud'].str.strip()
print("\nUnique values in 'isFraud' column after stripping:", df['isFraud'].unique())

df['isFraud'] = df['isFraud'].map({'TRUE': 1, 'FALSE': 0})

# Print unique values in 'isFraud' column after conversion
print("\nUnique values in 'isFraud' column after conversion:", df['isFraud'].unique())
print("Value counts for 'isFraud' after conversion:")
print(df['isFraud'].value_counts(dropna=False))

# If 'isFraud' column still contains NaN, fill with the most common value
if df['isFraud'].isnull().any():
    most_common = df['isFraud'].mode()[0]
    df['isFraud'] = df['isFraud'].fillna(most_common)
    print(f"\nFilled NaN values in 'isFraud' with the most common value: {most_common}")

print("\nDataset info after conversion and filling NaN:")
print(df[numeric_columns + ['isFraud']].info())

print("\nValue counts for 'isFraud':")
print(df['isFraud'].value_counts(normalize=True))

# Define features and target
features = ['WITHDRAWAL_AMT', 'DEPOSIT_AMT', 'BALANCE_AMT']
target = 'isFraud'

# Split the data
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a logistic regression model
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train_scaled, y_train)

# Save the model
joblib.dump(logreg, 'fraud_detection_model.joblib')

# Make predictions
y_pred_logreg = logreg.predict(X_test_scaled)

# Print classification report
print("\nLogistic Regression Classification Report:")
print(classification_report(y_test, y_pred_logreg))

# Evaluate the model
accuracy = logreg.score(X_test_scaled, y_test)
print(f"\nModel Accuracy: {accuracy:.2f}")

# Function to detect fraud
def detect_fraud(transaction):
    transaction = pd.DataFrame([transaction], columns=features)
    scaled_transaction = scaler.transform(transaction)
    prediction = logreg.predict(scaled_transaction)[0]
    return prediction == 1

# Test the function
test_transaction = {'WITHDRAWAL_AMT': 1000, 'DEPOSIT_AMT': 500, 'BALANCE_AMT': 10000}
result = detect_fraud(test_transaction)
print("\nFraud detection result:", "Fraud" if result else "Not Fraud")

# Print some statistics about the dataset
print("\nDataset Statistics:")
print(df[numeric_columns + ['isFraud']].describe())
print("\nFraud Percentage:", df['isFraud'].mean() * 100, "%")