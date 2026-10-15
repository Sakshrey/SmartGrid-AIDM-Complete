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
    "interarrival time",
    "time difference"
]

normal = fdi[fdi["attack_type"] == "Normal"].copy()
fdi_attack = fdi[fdi["attack_type"] == "FDI"].copy()
tsa_attack = tsa[tsa["attack_type"] == "TSA"].copy()

size = min(len(normal), len(fdi_attack), len(tsa_attack))

if size == 0:
    raise ValueError("One or more classes have no training samples.")

normal = normal.sample(size, random_state=42).copy()
fdi_attack = fdi_attack.sample(size, random_state=42).copy()
tsa_attack = tsa_attack.sample(size, random_state=42).copy()

np.random.seed(42)

# Generate subtle FDI samples
print("\nGenerating subtle FDI samples...")

subtle_fdi = normal.copy()

subtle_fdi["Actual frequency value"] += np.random.uniform(
    0.001, 0.005, len(subtle_fdi)
)

subtle_fdi["attack_type"] = "FDI"

# Generate synthetic TSA samples
print("Generating synthetic TSA samples...")

subtle_tsa = normal.copy()

# Shift PMU timestamps by five minutes and wrap at 3600 seconds
subtle_tsa["PMU time"] = (
    subtle_tsa["PMU time"] + 300
) % 3600

# Calculate time difference in minutes
subtle_tsa["time difference"] = (
    subtle_tsa["PDC time"] - subtle_tsa["PMU time"]
) / 60

# Correct for the timestamp cycle boundary
subtle_tsa["time difference"] = (
    (subtle_tsa["time difference"] + 30) % 60
) - 30

subtle_tsa["attack_type"] = "TSA"

# Combine training samples
train_data = pd.concat(
    [
        normal,
        fdi_attack,
        subtle_fdi,
        tsa_attack,
        subtle_tsa
    ],
    ignore_index=True
)

x_train = train_data[features].copy()
y_train = train_data["attack_type"].copy()

if x_train.isnull().any().any():
    raise ValueError("Missing values found in training features.")

if not np.isfinite(x_train.to_numpy(dtype=float)).all():
    raise ValueError("Invalid numeric values found in training features.")

print("\nTraining samples:", len(train_data))
print("\nClass distribution:")
print(y_train.value_counts())

print("\nTraining Random Forest...")

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
    "models/balanced_unified_random_forest.pkl"
)

print("\nTraining completed.")
print("Model saved successfully.")
print("Features:", model.n_features_in_)
print("Classes:", model.classes_)