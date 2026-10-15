
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

print("Loading autoencoder...")

autoencoder = load_model("models/autoencoder.keras")
scaler = joblib.load("models/autoencoder_scaler.pkl")

normal_data = pd.read_csv("dataset/processed_fdi_test.csv")
fdi_data = pd.read_csv("dataset/final_subtle_fdi_test.csv")
tsa_data = pd.read_csv("dataset/final_subtle_tsa_test.csv")

normal = normal_data[
    normal_data["attack_type"] == "Normal"
].sample(10000, random_state=42).copy()

fdi = fdi_data.sample(10000, random_state=42).copy()
tsa = tsa_data.sample(10000, random_state=42).copy()

normal["true_label"] = "Normal"
fdi["true_label"] = "FDI"
tsa["true_label"] = "TSA"

data = pd.concat([normal, fdi, tsa], ignore_index=True)

x = data.drop(
    columns=["attack_type", "true_label", "original_frequency"],
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

labels = data["true_label"].values

thresholds = [0.057621, 0.315663, 0.371082, 0.459954, 0.496289]

print("\nAutoencoder Threshold Comparison")
print("--------------------------------")

for threshold in thresholds:
    flagged = errors > threshold

    normal_mask = labels == "Normal"
    fdi_mask = labels == "FDI"
    tsa_mask = labels == "TSA"

    normal_false_alarms = np.mean(flagged[normal_mask]) * 100
    fdi_detection = np.mean(flagged[fdi_mask]) * 100
    tsa_detection = np.mean(flagged[tsa_mask]) * 100

    print("\nThreshold:", threshold)
    print("Normal false alarm rate:", round(normal_false_alarms, 2), "%")
    print("Subtle FDI flagged:", round(fdi_detection, 2), "%")
    print("Subtle TSA flagged:", round(tsa_detection, 2), "%")

print("\nEvaluation completed.")
