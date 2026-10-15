import pandas as pd
import numpy as np
import joblib

from tensorflow.keras.models import load_model
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

print("Loading models...")

model = joblib.load("models/balanced_unified_random_forest.pkl")
autoencoder = load_model("models/autoencoder.keras")
scaler = joblib.load("models/autoencoder_scaler.pkl")
threshold = joblib.load("models/autoencoder_threshold.pkl")

features = [
    "Actual frequency value",
    "Fraction of second",
    "PMU time",
    "PDC time",
    "interarrival time",
    "time difference"
]

print("Loading test datasets...")

fdi = pd.read_csv("dataset/processed_fdi_test.csv")
tsa = pd.read_csv("dataset/processed_tsa_test.csv")
subtle_fdi = pd.read_csv("dataset/final_subtle_fdi_test.csv")
subtle_tsa = pd.read_csv("dataset/final_subtle_tsa_test.csv")

normal = fdi[fdi["attack_type"] == "Normal"].sample(
    10000, random_state=42
).copy()

original_fdi = fdi[fdi["attack_type"] == "FDI"].sample(
    10000, random_state=42
).copy()

original_tsa = tsa[tsa["attack_type"] == "TSA"].sample(
    10000, random_state=42
).copy()

subtle_fdi = subtle_fdi.sample(
    10000, random_state=42
).copy()

subtle_tsa = subtle_tsa.sample(
    10000, random_state=42
).copy()

normal["true_label"] = "Normal"
original_fdi["true_label"] = "FDI"
subtle_fdi["true_label"] = "FDI"
original_tsa["true_label"] = "TSA"
subtle_tsa["true_label"] = "TSA"

test_data = pd.concat(
    [normal, original_fdi, subtle_fdi, original_tsa, subtle_tsa],
    ignore_index=True
)

x = test_data[features].copy()
true_label = test_data["true_label"].to_numpy()

print("\nDataset Verification")
print("--------------------")
print("Normal samples:", len(normal))
print("Original FDI samples:", len(original_fdi))
print("Subtle FDI samples:", len(subtle_fdi))
print("Original TSA samples:", len(original_tsa))
print("Subtle TSA samples:", len(subtle_tsa))
print("Feature shape:", x.shape)
print("Total test samples:", len(true_label))

print("\nTrue-label distribution:")
print(pd.Series(true_label).value_counts())

if x.isnull().any().any():
    raise ValueError("Missing values found in test features.")

if not np.isfinite(x.to_numpy(dtype=float)).all():
    raise ValueError("Non-finite values found in test features.")

if list(model.feature_names_in_) != features:
    print("\nWarning: Verify that test features match the model's training features.")

print("\nRunning Random Forest...")
rf_prediction = model.predict(x)

print("\nRunning Autoencoder...")

x_scaled = scaler.transform(x)

reconstructed = autoencoder.predict(
    x_scaled,
    batch_size=512,
    verbose=0
)

if reconstructed.shape != x_scaled.shape:
    raise ValueError(
        "Autoencoder output shape does not match scaled input shape."
    )

error = np.mean(
    np.square(x_scaled - reconstructed),
    axis=1
)

ae_prediction = error > threshold

print("\nAutoencoder Analysis")
print("--------------------")
print("Reconstruction error threshold:", threshold)
print("Autoencoder flagged samples:", int(np.sum(ae_prediction)))
print(
    "Autoencoder flagged percentage:",
    round(np.mean(ae_prediction) * 100, 2),
    "%"
)

print("\nFinal Prediction Selection")
print("--------------------------")
print("Using Random Forest predictions as the evaluation baseline.")

final_prediction = rf_prediction.copy()

print("Predictions changed by fusion: 0")

labels = ["Normal", "FDI", "TSA"]

accuracy = accuracy_score(true_label, final_prediction)

precision = precision_score(
    true_label,
    final_prediction,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    true_label,
    final_prediction,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    true_label,
    final_prediction,
    average="weighted",
    zero_division=0
)

cm = confusion_matrix(
    true_label,
    final_prediction,
    labels=labels
)

print("\nUnified AIDM Baseline Evaluation")
print("--------------------------------")
print("Total samples:", len(true_label))
print("Total predictions:", len(final_prediction))

print("\nOverall Results")
print("Accuracy :", round(accuracy * 100, 2), "%")
print("Precision:", round(precision * 100, 2), "%")
print("Recall   :", round(recall * 100, 2), "%")
print("F1 Score :", round(f1 * 100, 2), "%")

print("\nConfusion Matrix")
print("Rows = Actual labels")
print("Columns = Predicted labels")
print("Label order:", labels)
print(cm)

print("\nEvaluation Verification")
print("-----------------------")
print("Confusion matrix total:", cm.sum())
print("True-label count:", len(true_label))
print("Prediction count:", len(final_prediction))

print("\nClassification Report")
print("---------------------")

print(
    classification_report(
        true_label,
        final_prediction,
        labels=labels,
        zero_division=0
    )
)

print("\nClass-wise Detection")
print("--------------------")

for label in labels:
    mask = true_label == label
    total = int(np.sum(mask))
    correct = int(np.sum(final_prediction[mask] == label))

    detection_rate = correct / total * 100 if total else 0

    print(
        label,
        "correct:",
        correct,
        "out of",
        total
    )

    print(
        "Detection rate:",
        round(detection_rate, 2),
        "%"
    )

print("\nAttack-wise Detection")
print("---------------------")

groups = [
    ("Normal", 0, 10000),
    ("Original FDI", 10000, 20000),
    ("Subtle FDI", 20000, 30000),
    ("Original TSA", 30000, 40000),
    ("Subtle TSA", 40000, 50000)
]

for name, start, end in groups:
    actual = true_label[start:end]
    predicted = final_prediction[start:end]

    if len(actual) == 0:
        print(name, "- no samples")
        continue

    correct = int(np.sum(actual == predicted))
    total = len(actual)

    print(
        name + ":",
        round(correct / total * 100, 2),
        "% correctly classified"
    )

print("\nFalse Alarm Analysis")
print("--------------------")

normal_actual = true_label[:10000]
normal_predicted = final_prediction[:10000]

false_alarms = int(np.sum(normal_predicted != "Normal"))
normal_total = len(normal_actual)

print("Normal samples:", normal_total)
print("False alarms:", false_alarms)

if normal_total:
    print(
        "False Alarm Rate:",
        round(false_alarms / normal_total * 100, 2),
        "%"
    )

print("\nAutoencoder Analysis by Actual Class")
print("------------------------------------")

for name, start, end in groups:
    flags = ae_prediction[start:end]
    total = len(flags)
    flagged = int(np.sum(flags))

    print(
        name,
        "- flagged:",
        flagged,
        "out of",
        total,
        "(",
        round(flagged / total * 100, 2) if total else 0,
        "%)"
    )

print("\nPrediction Distribution")
print("-----------------------")
print(pd.Series(final_prediction).value_counts())

print("\nEvaluation completed.")