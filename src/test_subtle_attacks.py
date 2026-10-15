import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

model = joblib.load("models/unified_no_timediff_experimental.pkl")

features = [
    "Actual frequency value",
    "Fraction of second",
    "PMU time",
    "PDC time",
    "interarrival time"
]

tests = [
    ("Subtle FDI", "dataset/final_subtle_fdi_test.csv", "FDI"),
    ("Subtle TSA", "dataset/final_subtle_tsa_test.csv", "TSA")
]

for name, path, expected_class in tests:
    print("\n" + "=" * 50)
    print(name)
    print("=" * 50)

    data = pd.read_csv(path)

    print("Rows:", len(data))
    print("Columns:", list(data.columns))

    if "attack_type" in data.columns:
        y_true = data["attack_type"]
    else:
        y_true = pd.Series(
            [expected_class] * len(data),
            index=data.index
        )

    predictions = model.predict(data[features])

    print(
        "Accuracy:",
        round(accuracy_score(y_true, predictions) * 100, 2),
        "%"
    )

    print("\nConfusion Matrix (labels: FDI, Normal, TSA):")
    print(confusion_matrix(
        y_true,
        predictions,
        labels=["FDI", "Normal", "TSA"]
    ))

    print("\nClassification Report:")
    print(classification_report(
        y_true,
        predictions,
        labels=["FDI", "Normal", "TSA"],
        zero_division=0
    ))

    print("\nPredicted classes:")
    print(pd.Series(predictions).value_counts())