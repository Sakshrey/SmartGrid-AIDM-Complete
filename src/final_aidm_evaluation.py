import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix

fdi_model = joblib.load(
    "models/final_robust_random_forest.pkl"
)

tsa_model = joblib.load(
    "models/robust_tsa_random_forest.pkl"
)

autoencoder = load_model(
    "models/autoencoder.keras"
)

scaler = joblib.load(
    "models/autoencoder_scaler.pkl"
)

threshold = joblib.load(
    "models/autoencoder_threshold.pkl"
)

fdi_data = pd.read_csv(
    "dataset/processed_fdi_test.csv"
)

tsa_data = pd.read_csv(
    "dataset/processed_tsa_test.csv"
)

subtle_fdi = pd.read_csv(
    "dataset/final_subtle_fdi_test.csv"
)

subtle_tsa = pd.read_csv(
    "dataset/final_subtle_tsa_test.csv"
)

normal = fdi_data[
    fdi_data["attack_type"] == "Normal"
].copy()

original_fdi = fdi_data[
    fdi_data["attack_type"] == "FDI"
].copy()

original_tsa = tsa_data[
    tsa_data["attack_type"] == "TSA"
].copy()

normal = normal.sample(
    10000,
    random_state=42
)

original_fdi = original_fdi.sample(
    10000,
    random_state=42
)

original_tsa = original_tsa.sample(
    10000,
    random_state=42
)

subtle_fdi = subtle_fdi.sample(
    10000,
    random_state=42
)

subtle_tsa = subtle_tsa.sample(
    10000,
    random_state=42
)

normal["true_label"] = "Normal"
original_fdi["true_label"] = "FDI"
original_tsa["true_label"] = "TSA"
subtle_fdi["true_label"] = "FDI"
subtle_tsa["true_label"] = "TSA"

test_data = pd.concat(
    [
        normal,
        original_fdi,
        original_tsa,
        subtle_fdi,
        subtle_tsa
    ],
    ignore_index=True
)

x = test_data.drop(
    columns=[
        "attack_type",
        "true_label",
        "original_frequency"
    ],
    errors="ignore"
)

true_label = test_data["true_label"]

fdi_prediction = fdi_model.predict(x)
tsa_prediction = tsa_model.predict(x)

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
    "Attack",
    "Normal"
)

final_prediction = []

for i in range(len(x)):

    if fdi_prediction[i] == "FDI":
        final_prediction.append("FDI")

    elif tsa_prediction[i] == "TSA":
        final_prediction.append("TSA")

    elif ae_prediction[i] == "Attack":
        if fdi_prediction[i] == "FDI":
            final_prediction.append("FDI")
        else:
            final_prediction.append("TSA")

    else:
        final_prediction.append("Normal")

print("Final AIDM Evaluation")
print("---------------------")

print("Normal samples:", sum(true_label == "Normal"))
print("FDI samples:", sum(true_label == "FDI"))
print("TSA samples:", sum(true_label == "TSA"))
print("Total samples:", len(true_label))

print("\nOverall Results")

accuracy = accuracy_score(
    true_label,
    final_prediction
)

precision = precision_score(
    true_label,
    final_prediction,
    average="weighted"
)

recall = recall_score(
    true_label,
    final_prediction,
    average="weighted"
)

f1 = f1_score(
    true_label,
    final_prediction,
    average="weighted"
)

print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)

print("\nConfusion Matrix")
print(confusion_matrix(
    true_label,
    final_prediction,
    labels=["Normal", "FDI", "TSA"]
))

print("\nAttack-wise Detection")

for attack in ["FDI", "TSA"]:

    total = sum(true_label == attack)

    detected = sum(
        (true_label == attack) &
        (np.array(final_prediction) == attack)
    )

    rate = detected / total * 100

    print(
        attack,
        "Detection Rate:",
        rate,
        "%"
    )

print("\nPrediction Distribution")
print(
    pd.Series(final_prediction).value_counts()
)