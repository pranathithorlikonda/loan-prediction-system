import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
import joblib
import shap

# ---------------------------
# LOAD DATA
# ---------------------------
base_path = os.path.dirname(os.path.dirname(__file__))
file_path = os.path.join(base_path, 'data', 'loan_approval_dataset.csv')

df = pd.read_csv(file_path)

# ---------------------------
# CLEANING
# ---------------------------
df.columns = df.columns.str.strip()

df['loan_status'] = df['loan_status'].str.strip()
df['education'] = df['education'].str.strip()
df['self_employed'] = df['self_employed'].str.strip()

df = df.drop('loan_id', axis=1)

# Encode categorical
le = LabelEncoder()
df['education'] = le.fit_transform(df['education'])
df['self_employed'] = le.fit_transform(df['self_employed'])

# Target
df['loan_status'] = df['loan_status'].map({
    'Approved': 1,
    'Rejected': 0
})

# ---------------------------
# FEATURES
# ---------------------------
X = df.drop('loan_status', axis=1)
y = df['loan_status']

feature_names = X.columns.tolist()

# ---------------------------
# SPLIT
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------
# SCALING
# ---------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------------------------
# MODEL
# ---------------------------
model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)

model.fit(X_train, y_train)

# ---------------------------
# EVALUATION
# ---------------------------
y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ---------------------------
# SHAP EXPLAINER
# ---------------------------
explainer = shap.Explainer(model)

# Save a sample SHAP explanation (optional test)
shap_values = explainer(X_test[:1])

print("\nSample SHAP values:", shap_values.values)

# ---------------------------
# SAVE EVERYTHING
# ---------------------------
joblib.dump(model, os.path.join(base_path, 'ml', 'model.pkl'))
joblib.dump(scaler, os.path.join(base_path, 'ml', 'scaler.pkl'))
joblib.dump(explainer, os.path.join(base_path, 'ml', 'shap_explainer.pkl'))
joblib.dump(feature_names, os.path.join(base_path, 'ml', 'features.pkl'))
joblib.dump(X_test, os.path.join(base_path, 'ml', 'X_test.pkl'))


print("\n✅ Model, Scaler, SHAP Explainer Saved Successfully!")