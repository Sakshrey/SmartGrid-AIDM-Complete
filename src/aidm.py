import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

rf_model = joblib.load("models/final_robust_random_forest.pkl")

autoencoder = load_model("models/autoencoder.keras")

scaler = joblib.load("models/autoencoder_scaler.pkl")
threshold = joblib.load("models/autoencoder_threshold.pkl")

data = pd.read_csv("dataset/subtle_fdi_attack.csv")

X = data.drop(columns=["attack_type", "original_frequency"])

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

y = ["FDI"] * len(data)

detected = sum(final_prediction == "FDI")
missed = sum(final_prediction == "Normal")

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

print("AIDM Fusion Results")
print("-------------------")

print("Total attacked samples:", len(data))

print("\nRobust Random Forest:")
print("Detected:", sum(rf_prediction == "FDI"))

print("\nAutoencoder:")
print("Detected:", sum(ae_prediction == "FDI"))

print("\nFinal AIDM:")
print("Detected as FDI:", detected)
print("Detected as Normal:", missed)

print("\nDetection Rate:", recall * 100, "%")
print("Miss Rate:", (1 - recall) * 100, "%")

print("\nPrecision:", precision)
print("F1 Score:", f1)

print("\nConfusion Matrix:")
print(confusion_matrix(
    y,
    final_prediction,
    labels=["Normal", "FDI"]
))