import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

features_6 = [
    "Actual frequency value",
    "Fraction of second",
    "PMU time",
    "PDC time",
    "interarrival time",
    "time difference"
]

features_5 = [
    "Actual frequency value",
    "Fraction of second",
    "PMU time",
    "PDC time",
    "interarrival time"
]

print("Loading test datasets...")

normal_fdi = pd.read_csv("dataset/processed_fdi_test.csv")
tsa_data = pd.read_csv("dataset/processed_tsa_test.csv")

normal = normal_fdi[
    normal_fdi["attack_type"] == "Normal"
].sample(n=5000, random_state=42)

fdi = normal_fdi[
    normal_fdi["attack_type"] == "FDI"
].sample(n=5000, random_state=42)

tsa = tsa_data[
    tsa_data["attack_type"] == "TSA"
].sample(n=5000, random_state=42)

test_data = pd.concat([normal, fdi, tsa], ignore_index=True)
y_test = test_data["attack_type"]

models = [
    (
        "Original balanced model",
        "models/balanced_unified_random_forest.pkl",
        features_6
    ),
    (
        "Experimental no-time-difference model",
        "models/unified_no_timediff_experimental.pkl",
        features_5
    )
]

labels = ["Normal", "FDI", "TSA"]

for name, path, features in models:
    print("\n" + "=" * 55)
    print(name)
    print("=" * 55)

    model = joblib.load(path)
    predictions = model.predict(test_data[features])

    print(
        "Accuracy:",
        round(accuracy_score(y_test, predictions) * 100, 2),
        "%"
    )

    print("\nConfusion Matrix (rows = actual, columns = predicted):")
    print(confusion_matrix(y_test, predictions, labels=labels))

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        labels=labels,
        zero_division=0
    ))

    normal_predictions = predictions[:5000]
    false_alarms = sum(normal_predictions != "Normal")

    print(
        "Normal false-alarm rate:",
        round(false_alarms / 5000 * 100, 2),
        "%"
    )