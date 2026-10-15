
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

print("Loading autoencoder...")

autoencoder = load_model("models/autoencoder.keras")
scaler = joblib.load("models/autoencoder_scaler.pkl")
threshold = joblib.load("models/autoencoder_threshold.pkl")

print("Loading Normal validation samples...")

data = pd.read_csv("dataset/processed_fdi_test.csv")
normal = data[data["attack_type"] == "Normal"].copy()

x = normal.drop(
    columns=["attack_type"],
    errors="ignore"
)

x_scaled = scaler.transform(x)

reconstructed = autoencoder.predict(
    x_scaled,
    batch_size=512,
    verbose=0
)

errors = np.mean(
    np.square(x_scaled - reconstructed),
    axis=1
)

print("\nAutoencoder Threshold Evaluation")
print("--------------------------------")
print("Samples tested:", len(errors))
print("Current threshold:", threshold)
print("Mean reconstruction error:", round(float(np.mean(errors)), 6))
print("Median reconstruction error:", round(float(np.median(errors)), 6))
print("95th percentile:", round(float(np.percentile(errors, 95)), 6))
print("99th percentile:", round(float(np.percentile(errors, 99)), 6))

print("\nFalse alarms at different thresholds")
print("------------------------------------")

for percentile in [95, 97, 99, 99.5]:
    new_threshold = np.percentile(errors, percentile)
    flagged = np.sum(errors > new_threshold)
    rate = flagged / len(errors) * 100

    print(
        f"{percentile}th percentile threshold:",
        round(float(new_threshold), 6),
        "| False alarms:",
        int(flagged),
        "| Rate:",
        round(float(rate), 2),
        "%"
    )

print("\nEvaluation completed.")
