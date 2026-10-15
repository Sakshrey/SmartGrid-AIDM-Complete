
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

model = joblib.load("models/balanced_unified_random_forest.pkl")

fdi = pd.read_csv("dataset/processed_fdi_test.csv")
tsa = pd.read_csv("dataset/processed_tsa_test.csv")

fdi = fdi[fdi["attack_type"] != "Normal"].copy()
tsa = tsa[tsa["attack_type"] != "Normal"].copy()

fdi["true_label"] = "FDI"
tsa["true_label"] = "TSA"

data = pd.concat([fdi, tsa], ignore_index=True)

features = [
    "Actual frequency value",
    "Fraction of second",
    "PMU time",
    "PDC time",
    "interarrival time",
    "time difference"
]

X = data[features]
y = data["true_label"]

predictions = model.predict(X)

labels = ["FDI", "TSA"]

print("\nOriginal Attacks Evaluation")
print("===========================")

print("Total test samples:", len(data))
print("FDI samples:", sum(y == "FDI"))
print("TSA samples:", sum(y == "TSA"))

print("\nOverall Accuracy:")
print(f"{accuracy_score(y, predictions) * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(
    y,
    predictions,
    labels=labels,
    target_names=labels,
    zero_division=0
))

print("Confusion Matrix (Actual rows, Predicted columns)")
print("                 Predicted FDI   Predicted TSA   Predicted Normal")

cm = confusion_matrix(
    y,
    predictions,
    labels=["FDI", "TSA", "Normal"]
)

print(f"Actual FDI       {cm[0][0]:14d}   {cm[0][1]:14d}   {cm[0][2]:16d}")
print(f"Actual TSA       {cm[1][0]:14d}   {cm[1][1]:14d}   {cm[1][2]:16d}")

print("\nPer-Class Detection Rates:")

for label in labels:
    total = sum(y == label)
    correct = sum((y == label) & (predictions == label))
    rate = correct / total * 100 if total else 0

    print(f"{label}: {correct}/{total} = {rate:.2f}%")

print("\nEvaluation completed.")
