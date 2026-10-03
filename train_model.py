
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

DATA_PATH = os.path.join("data", "diabetes.csv")
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]

# If you have the real Pima Indians Diabetes Dataset, save it as:
# data/diabetes.csv
# with columns exactly matching FEATURES + Outcome.
#
# The included demo CSV is only for software execution/testing.
df = pd.read_csv(DATA_PATH)

missing = [c for c in FEATURES + ["Outcome"] if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns in dataset: {missing}")

X = df[FEATURES].copy()
y = df["Outcome"].astype(int)

# In Pima data, zeros in some clinical columns are commonly treated as
# missing/invalid measurements for preprocessing purposes.
zero_as_missing = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
X[zero_as_missing] = X[zero_as_missing].replace(0, np.nan)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

imputer = SimpleImputer(strategy="median")
scaler = StandardScaler()

X_train_imp = imputer.fit_transform(X_train)
X_test_imp = imputer.transform(X_test)

X_train_scaled = scaler.fit_transform(X_train_imp)
X_test_scaled = scaler.transform(X_test_imp)

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    min_samples_split=4,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train_scaled, y_train)

pred = model.predict(X_test_scaled)

print("\n===== VitalPredict Random Forest Evaluation =====")
print(f"Accuracy : {accuracy_score(y_test, pred):.4f}")
print(f"Precision: {precision_score(y_test, pred, zero_division=0):.4f}")
print(f"Recall   : {recall_score(y_test, pred, zero_division=0):.4f}")
print(f"F1 Score : {f1_score(y_test, pred, zero_division=0):.4f}")

# Save the fitted preprocessing objects separately.
# app.py uses the scaler; imputation is repeated below through saved medians.
joblib.dump(model, os.path.join(MODEL_DIR, "random_forest.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
joblib.dump(imputer, os.path.join(MODEL_DIR, "imputer.pkl"))
joblib.dump(FEATURES, os.path.join(MODEL_DIR, "features.pkl"))

# Save medians so the Flask app can preprocess user input consistently.
joblib.dump(imputer.statistics_, os.path.join(MODEL_DIR, "imputer_medians.pkl"))

print("\nModel files created inside /model")
