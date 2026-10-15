import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

print("Loading training data...")

fdi = pd.read_csv("dataset/processed_fdi.csv")
tsa = pd.read_csv("dataset/processed_tsa.csv")

features = [
    "Actual frequency value",
    "Fraction of second",
    "PMU time",
    "PDC time",
    "interarrival time"
]

normal = fdi[fdi["attack_type"] == "Normal"].copy()
fdi_attack = fdi[fdi["attack_type"] == "FDI"].copy()
tsa_attack = tsa[tsa["attack_type"] == "TSA"].copy()

size = min(len(normal), len(fdi_attack), len(tsa_attack))

normal = normal.sample(size, random_state=42)
fdi_attack = fdi_attack.sample(size, random_state=42)
tsa_attack = tsa_attack.sample(size, random_state=42)

train = pd.concat(
    [normal, fdi_attack, tsa_attack],
    ignore_index=True
)

x_train = train[features]
y_train = train["attack_type"]

print("Training model without time difference...")

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(x_train, y_train)

print("Loading test data...")

fdi_test = pd.read_csv("dataset/processed_fdi_test.csv")
tsa_test = pd.read_csv("dataset/processed_tsa_test.csv")
subtle_fdi = pd.read_csv("dataset/final_subtle_fdi_test.csv")
subtle_tsa = pd.read_csv("dataset/final_subtle_tsa_test.csv")

normal_test = fdi_test[fdi_test["attack_type"] == "Normal"].sample(
    10000, random_state=42
).copy()

original_fdi = fdi_test[fdi_test["attack_type"] == "FDI"].sample(
    10000, random_state=42
).copy()

original_tsa = tsa_test[tsa_test["attack_type"] == "TSA"].sample(
    10000, random_state=42
).copy()

subtle_fdi = subtle_fdi.sample(10000, random_state=42).copy()
subtle_tsa = subtle_tsa.sample(10000, random_state=42).copy()

normal_test["true_label"] = "Normal"
original_fdi["true_label"] = "FDI"
original_tsa["true_label"] = "TSA"
subtle_fdi["true_label"] = "FDI"
subtle_tsa["true_label"] = "TSA"

test = pd.concat(
    [normal_test, original_fdi, subtle_fdi, original_tsa, subtle_tsa],
    ignore_index=True
)

x_test = test[features]
y_test = test["true_label"]

predictions = model.predict(x_test)

print("\nResults without time difference")
print("-------------------------------")
print("Accuracy:", round(accuracy_score(y_test, predictions) * 100, 2), "%")
print(classification_report(y_test, predictions, zero_division=0))