import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

rf_model = joblib.load("models/final_robust_random_forest.pkl")

autoencoder = load_model("models/autoencoder.keras")

scaler = joblib.load("models/autoencoder_scaler.pkl")
threshold = joblib.load("models/autoencoder_threshold.pkl")

data = pd.read_csv("dataset/processed_fdi.csv")

normal_data = data[data["attack_type"] == "Normal"].copy()
fdi_data = data[data["attack_type"] == "FDI"].copy()

normal_test = normal_data.sample(
    n=min(10000, len(normal_data)),
    random_state=42
)

fdi_test = fdi_data.sample(
    n=min(10000, len(fdi_data)),
    random_state=42
)

subtle_attack = normal_test.copy()

np.random.seed(42)

subtle_attack["Actual frequency value"] = (
    subtle_attack["Actual frequency value"] +
    np.random.uniform(0.001, 0.005, len(subtle_attack))
)

subtle_attack["attack_type"] = "FDI"

test_data = pd.concat(
    [normal_test, fdi_test, subtle_attack],
    ignore_index=True
)

X = test_data.drop(columns=["attack_type"])
y = test_data["attack_type"]

rf_prediction = rf_model.predict(X)

X_scaled = scaler.transform(X)

reconstructed = autoencoder.predict(
    X_scaled,
    verbose=0
)

errors = np.mean(
    np.square(X_scaled - reconstructed),
    axis=1
)

ae_prediction = np.where(
    errors > threshold,
    "FDI",
    "Normal"
)

final_prediction = np.where(
    (rf_prediction == "FDI") | (ae_prediction == "FDI"),
    "FDI",
    "Normal"
)

accuracy = accuracy_score(y, final_prediction)

precision = precision_score(
    y,
    final_prediction,
    pos_label="FDI",
    zero_division=0
)

recall = recall_score(
    y,
    final_prediction,
    pos_label="FDI",
    zero_division=0
)

f1 = f1_score(
    y,
    final_prediction,
    pos_label="FDI",
    zero_division=0
)

print("FINAL AIDM EVALUATION")
print("---------------------")

print("Normal samples:", len(normal_test))
print("Original FDI samples:", len(fdi_test))
print("Subtle FDI samples:", len(subtle_attack))
print("Total test samples:", len(test_data))

print("\nAccuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)

print("\nConfusion Matrix:")
print(confusion_matrix(
    y,
    final_prediction,
    labels=["Normal", "FDI"]
))

print("\nPrediction Distribution:")
print(pd.Series(final_prediction).value_counts())