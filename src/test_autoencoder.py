import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

model = load_model("models/autoencoder.keras")

scaler = joblib.load("models/autoencoder_scaler.pkl")
threshold = joblib.load("models/autoencoder_threshold.pkl")

data = pd.read_csv("dataset/subtle_fdi_attack.csv")

X = data.drop(columns=["attack_type", "original_frequency"])

X = scaler.transform(X)

reconstructed = model.predict(X, verbose=0)

errors = np.mean(
    np.square(X - reconstructed),
    axis=1
)

prediction = np.where(
    errors > threshold,
    "FDI",
    "Normal"
)

detected = sum(prediction == "FDI")
missed = sum(prediction == "Normal")

print("Autoencoder - Subtle FDI Attack")
print("--------------------------------")

print("Total attacked samples:", len(data))
print("Detected as FDI:", detected)
print("Detected as Normal:", missed)

print("\nDetection Rate:", detected / len(data) * 100, "%")
print("Miss Rate:", missed / len(data) * 100, "%")

print("\nAnomaly Threshold:", threshold)

print("\nPrediction Distribution:")
print(pd.Series(prediction).value_counts())