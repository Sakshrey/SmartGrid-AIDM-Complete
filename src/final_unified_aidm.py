import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix

model = joblib.load(
    "models/balanced_unified_random_forest.pkl"
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

fdi = pd.read_csv(
    "dataset/processed_fdi_test.csv"
)

tsa = pd.read_csv(
    "dataset/processed_tsa_test.csv"
)

subtle_fdi = pd.read_csv(
    "dataset/final_subtle_fdi_test.csv"
)

subtle_tsa = pd.read_csv(
    "dataset/final_subtle_tsa_test.csv"
)

normal = fdi[
    fdi["attack_type"] == "Normal"
].sample(10000, random_state=42)

original_fdi = fdi[
    fdi["attack_type"] == "FDI"
].sample(10000, random_state=42)

original_tsa = tsa[
    tsa["attack_type"] == "TSA"
].sample(10000, random_state=42)

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
        subtle_fdi,
        original_tsa,
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

true_label = test_data["true_label"].values

print("Running Random Forest...")

rf_prediction = model.predict(x)

print("Running Autoencoder...")

x_scaled = scaler.transform(x)

reconstructed = autoencoder.predict(
    x_scaled,
    batch_size=512,
    verbose=0
)

error = np.mean(
    np.square(x_scaled - reconstructed),
    axis=1
)

ae_prediction = error > threshold

print("Combining AIDM results...")

final_prediction = rf_prediction.copy()

rf_probability = model.predict_proba(x)

classes = model.classes_

for i in range(len(final_prediction)):

    if ae_prediction[i] and final_prediction[i] == "Normal":

        best_index = np.argmax(
            rf_probability[i]
        )

        final_prediction[i] = classes[best_index]

print("\nFinal Unified AIDM Evaluation")
print("------------------------------")

print("Normal samples:", sum(true_label == "Normal"))
print("FDI samples:", sum(true_label == "FDI"))
print("TSA samples:", sum(true_label == "TSA"))
print("Total samples:", len(true_label))

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

print("\nOverall Results")

print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)

print("\nConfusion Matrix")

print(
    confusion_matrix(
        true_label,
        final_prediction,
        labels=[
            "Normal",
            "FDI",
            "TSA"
        ]
    )
)

print("\nAttack-wise Detection")

for attack in ["FDI", "TSA"]:

    total = sum(
        true_label == attack
    )

    detected = sum(
        (true_label == attack) &
        (final_prediction == attack)
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
    pd.Series(
        final_prediction
    ).value_counts()
)