import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

fdi = pd.read_csv("dataset/processed_fdi.csv")
tsa = pd.read_csv("dataset/processed_tsa.csv")

normal = fdi[
    fdi["attack_type"] == "Normal"
].copy()

fdi_attack = fdi[
    fdi["attack_type"] == "FDI"
].copy()

tsa_attack = tsa[
    tsa["attack_type"] == "TSA"
].copy()

normal["attack_type"] = "Normal"

fdi_attack["attack_type"] = "FDI"

tsa_attack["attack_type"] = "TSA"

np.random.seed(42)

subtle_fdi = normal.copy()

subtle_fdi["Actual frequency value"] += np.random.uniform(
    0.001,
    0.005,
    len(subtle_fdi)
)

subtle_fdi["attack_type"] = "FDI"

subtle_tsa = normal.copy()

subtle_tsa["Actual frequency value"] += np.random.uniform(
    0.001,
    0.005,
    len(subtle_tsa)
)

subtle_tsa["attack_type"] = "TSA"

train_data = pd.concat(
    [
        normal,
        fdi_attack,
        tsa_attack,
        subtle_fdi,
        subtle_tsa
    ],
    ignore_index=True
)

x_train = train_data.drop(
    columns=["attack_type"]
)

y_train = train_data["attack_type"]

print("Unified Robust IDS")
print("------------------")

print("Training samples:", len(x_train))

print("\nClass distribution:")
print(y_train.value_counts())

model = RandomForestClassifier(
    n_estimators=30,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)

model.fit(
    x_train,
    y_train
)

joblib.dump(
    model,
    "models/unified_robust_random_forest.pkl"
)

print("\nUnified model trained successfully")
print("Model saved successfully")