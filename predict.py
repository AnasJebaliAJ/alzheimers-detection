import joblib
import pandas as pd
from pathlib import Path


MODEL_PATH = Path(__file__).resolve().parent / "models" / "logistic_regression.joblib"

model = joblib.load(MODEL_PATH)

FEATURES = [
    "M/F", "Age", "EDUC", "SES",
    "MMSE", "eTIV", "nWBV", "ASF"
]



def validate_patient(patient_data):
    if not isinstance(patient_data, dict):
        raise TypeError("Patient data must be a dictionary")

    missing = [
        feature for feature in FEATURES
        if feature not in patient_data
    ]

    if missing:
        raise ValueError(f"Missing features: {missing}")

    if patient_data["M/F"] not in ["M", "F"]:
        raise ValueError("M/F must be 'M' or 'F'")

    if not 0 <= patient_data["Age"] <= 120:
        raise ValueError("Age must be between 0 and 120")


def predict_patient(patient_data):
    # Validate before prediction
    validate_patient(patient_data)

    # Convert input dictionary into a DataFrame
    patient = pd.DataFrame([patient_data])

    # Generate prediction and probabilities
    prediction = model.predict(patient)[0]
    probabilities = model.predict_proba(patient)[0]

    labels = {
        0: "Nondemented",
        1: "Demented"
    }

    return {
        "prediction": labels[prediction],
        "probabilities": {
            labels[class_id]: float(probability)
            for class_id, probability in zip(
                model.classes_, probabilities
            )
        }
    }



if __name__ == "__main__":
    sample = {
        "M/F": "F",
        "Age": 75,
        "EDUC": 12,
        "SES": 2.0,
        "MMSE": 27.0,
        "eTIV": 1500,
        "nWBV": 0.72,
        "ASF": 1.17
    }

    result = predict_patient(sample)

    print("\n MODEL PREDICTION ")
    print("Predicted class:", result["prediction"])

    for label, probability in result["probabilities"].items():
        print(f"{label}: {probability:.2%}")
