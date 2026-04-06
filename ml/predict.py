import joblib
import numpy as np
import os

# ---------------------------
# LOAD FILES
# ---------------------------
base_path = os.path.dirname(__file__)

model = joblib.load(os.path.join(base_path, 'model.pkl'))
scaler = joblib.load(os.path.join(base_path, 'scaler.pkl'))
explainer = joblib.load(os.path.join(base_path, 'shap_explainer.pkl'))
feature_names = joblib.load(os.path.join(base_path, 'features.pkl'))
X_test = joblib.load(os.path.join(base_path, 'X_test.pkl'))

# ---------------------------
# FEATURE NAME MAPPING (USER FRIENDLY)
# ---------------------------
feature_map = {
    "no_of_dependents": "Number of Dependents",
    "education": "Education",
    "self_employed": "Self Employed",
    "income_annum": "Annual Income",
    "loan_amount": "Loan Amount",
    "loan_term": "Loan Term",
    "cibil_score": "CIBIL Score",
    "residential_assets_value": "Residential Assets",
    "commercial_assets_value": "Commercial Assets",
    "luxury_assets_value": "Luxury Assets",
    "bank_asset_value": "Bank Balance"
}

# ---------------------------
# EXPLANATION FUNCTION
# ---------------------------
def generate_explanation(shap_values):
    positive_reasons = []
    negative_reasons = []

    for feature, value in shap_values.items():
        name = feature_map.get(feature, feature)

        if value > 0:
            positive_reasons.append(f"{name} increased approval chances")
        elif value < 0:
            negative_reasons.append(f"{name} reduced approval chances")

    return positive_reasons, negative_reasons


# ---------------------------
# PREDICTION FUNCTION
# ---------------------------
def predict_loan(data, use_scaling=True):

    data = np.array(data).reshape(1, -1)

    # Apply scaling only for new input
    if use_scaling:
        data = scaler.transform(data)

    # Prediction
    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]

    # SHAP values
    shap_values = explainer(data)

    shap_result = {}
    for i, val in enumerate(shap_values.values[0]):
        shap_result[feature_names[i]] = float(val)

    # Generate explanations
    positive, negative = generate_explanation(shap_result)

    # Decision logic
    if probability > 0.75:
        decision = "Approved ✅"
    elif probability > 0.50:
        decision = "Manual Review ⚠️"
    else:
        decision = "Rejected ❌"

    return {
        "prediction": int(prediction),
        "probability": float(probability),
        "decision": decision,
        "positive_reasons": positive,
        "negative_reasons": negative,
        "shap_values": shap_result
    }


# ---------------------------
# TEST USING X_test
# ---------------------------
if __name__ == "__main__":

    print("\n🔍 Testing with X_test sample")

    sample = X_test[0]

    result = predict_loan(sample, use_scaling=False)

    print("\n✅ Prediction Result:")
    print("Prediction:", result["prediction"])
    print("Probability:", result["probability"])
    print("Decision:", result["decision"])

    print("\n📊 Positive Reasons:")
    for r in result["positive_reasons"]:
        print("✔", r)

    print("\n📊 Negative Reasons:")
    for r in result["negative_reasons"]:
        print("❌", r)