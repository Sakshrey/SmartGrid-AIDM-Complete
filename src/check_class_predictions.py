
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report

model = joblib.load("models/balanced_unified_random_forest.pkl")

fdi = pd.read_csv("dataset/processed_fdi_test.csv")
tsa = pd.read_csv("dataset/processed_tsa_test.csv")

fdi_normal = fdi[fdi["attack_type"] == "Normal"].copy()
fdi_attack = fdi[fdi["attack_type"] != "Normal"].copy()
tsa_attack = tsa[tsa["attack_type"] != "Normal"].copy()

fdi_normal["true_label"] = "Normal"
fdi_attack["true_label"] = "FDI"
tsa_attack["true_label"] = "TSA"

data = pd.concat(
    [fdi_normal, fdi_attack, tsa_attack],
    ignore_index=True
)

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

print("\nOriginal Test Data Evaluation")
print("=============================")
print("Total samples:", len(data))
print("Overall accuracy:", round(accuracy_score(y, predictions) * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(
    y,
    predictions,
    labels=["Normal", "FDI", "TSA"],
    target_names=["Normal", "FDI", "TSA"],
    zero_division=0
))

print("\nActual vs Predicted Counts:")
results = pd.DataFrame({
    "Actual": y,
    "Predicted": predictions
})

print(pd.crosstab(
    results["Actual"],
    results["Predicted"],
    rownames=["Actual"],
    colnames=["Predicted"],
    dropna=False
))

