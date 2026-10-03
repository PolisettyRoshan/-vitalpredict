
from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import os
from datetime import datetime

app = Flask(__name__)

MODEL_PATH = os.path.join("model", "random_forest.pkl")
SCALER_PATH = os.path.join("model", "scaler.pkl")
FEATURES_PATH = os.path.join("model", "features.pkl")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
IMPUTER_PATH = os.path.join("model", "imputer.pkl")
imputer = joblib.load(IMPUTER_PATH)
FEATURES = joblib.load(FEATURES_PATH)

history = []

def risk_label(probability):
    # Demo screening bands for this academic project.
    if probability < 0.35:
        return "Low Risk"
    return "Medium Risk"

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)

        row = {
            "Pregnancies": float(data["pregnancies"]),
            "Glucose": float(data["glucose"]),
            "BloodPressure": float(data["blood_pressure"]),
            "SkinThickness": float(data["skin_thickness"]),
            "Insulin": float(data["insulin"]),
            "BMI": float(data["bmi"]),
            "DiabetesPedigreeFunction": float(data["pedigree"]),
            "Age": float(data["age"])
        }

        X = pd.DataFrame([row], columns=FEATURES)
        zero_as_missing = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
        X[zero_as_missing] = X[zero_as_missing].replace(0, float("nan"))
        X_imputed = imputer.transform(X)
        X_scaled = scaler.transform(X_imputed)

        probability = float(model.predict_proba(X_scaled)[0][1])
        prediction = int(model.predict(X_scaled)[0])
        label = risk_label(probability)

        importances = dict(zip(FEATURES, model.feature_importances_))
        top_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5]

        result = {
            "prediction": prediction,
            "risk": label,
            "probability": round(probability * 100, 2),
            "top_features": [
                {"feature": f, "importance": round(v * 100, 2)}
                for f, v in top_features
            ],
            "timestamp": datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        }

        history.insert(0, result)
        if len(history) > 10:
            history.pop()

        return jsonify({"success": True, "result": result})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

@app.route("/history")
def get_history():
    return jsonify(history)

@app.route("/health")
def health():
    return jsonify({"status": "VitalPredict backend is running"})

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)