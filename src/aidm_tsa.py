import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

data = pd.read_csv("dataset/subtle_tsa_attack.csv")

rf_model = joblib.load("models/robust_tsa_random_forest.pkl")

autoencoder = load_model("models/autoencoder.keras")
scaler = joblib.load("models/autoencoder_scaler.pkl")
threshold = joblib.load("models/autoencoder_threshold.pkl")

x = data.drop(columns=[
    "attack_type",
    "original_frequency"
])

rf_prediction = rf_model.predict(x)

x_scaled = scaler.transform(x)

reconstructed = autoencoder.predict(
    x_scaled,
    verbose=0
)

error = np.mean(
    np.square(x_scaled - reconstructed),
    axis=1
)

ae_prediction = np.where(
    error > threshold,
    "TSA",
    "Normal"
)

final_prediction = np.where(
    (rf_prediction == "TSA") | (ae_prediction == "TSA"),
    "TSA",
    "Normal"
)

rf_detected = sum(rf_prediction == "TSA")
ae_detected = sum(ae_prediction == "TSA")
final_detected = sum(final_prediction == "TSA")

print("Total attacked samples:", len(data))

print("Robust RF detected:", rf_detected)
print("Autoencoder detected:", ae_detected)
print("Final AIDM detected:", final_detected)

print("Normal:", len(data) - final_detected)

detection_rate = final_detected / len(data) * 100
miss_rate = 100 - detection_rate

print("\nDetection Rate:", detection_rate, "%")
print("Miss Rate:", miss_rate, "%")