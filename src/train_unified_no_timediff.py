import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

print("Loading datasets...")

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

normal = normal.sample(size, random_state=42).copy()
fdi_attack = fdi_attack.sample(size, random_state=42).copy()
tsa_attack = tsa_attack.sample(size, random_state=42).copy()

np.random.seed(42)

print("Generating subtle FDI samples...")

subtle_fdi = normal.copy()

subtle_fdi["Actual frequency value"] += np.random.uniform(
    0.001, 0.005, len(subtle_fdi)
)

subtle_fdi["attack_type"] = "FDI"

print("Generating synthetic TSA samples...")

subtle_tsa = normal.copy()

subtle_tsa["PMU time"] = (
    subtle_tsa["PMU time"] + 300
) % 3600

subtle_tsa["attack_type"] = "TSA"

train_data = pd.concat(
    [normal, fdi_attack, subtle_fdi, tsa_attack, subtle_tsa],
    ignore_index=True
)

x_train = train_data[features].copy()
y_train = train_data["attack_type"].copy()

print("\nTraining samples:", len(train_data))
print(y_train.value_counts())

print("\nTraining experimental model...")

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(x_train, y_train)

joblib.dump(
    model,
    "models/unified_no_timediff_experimental.pkl"
)

print("\nExperimental model saved.")
print("Features:", list(model.feature_names_in_))
print("Classes:", model.classes_)