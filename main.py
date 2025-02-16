import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the uploaded dataset
file_path = "StudentsPerformance.csv"
df = pd.read_csv(file_path)

# Selecting features and target
X = df[['math score', 'reading score', 'writing score']]
y = df['race/ethnicity']

# Encode target labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model and scaler
with open("model/student_performance_model.pkl", "wb") as f:
    pickle.dump((scaler, label_encoder, model), f)

print("✅ Model training completed and saved.")
