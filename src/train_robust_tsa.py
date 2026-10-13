import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv("dataset/processed_tsa.csv")

train_data, test_data = train_test_split(
    data,
    test_size=0.2,
    random_state=42,
    stratify=data["attack_type"]
)

normal_data = train_data[train_data["attack_type"] == "Normal"].copy()

np.random.seed(42)

normal_data["Actual frequency value"] += np.random.uniform(
    0.001,
    0.005,
    len(normal_data)
)

normal_data["attack_type"] = "TSA"

train_data = pd.concat(
    [train_data, normal_data],
    ignore_index=True
)

x_train = train_data.drop(columns=["attack_type"])
y_train = train_data["attack_type"]

x_test = test_data.drop(columns=["attack_type"])
y_test = test_data["attack_type"]

model = RandomForestClassifier(
    n_estimators=30,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)

model.fit(x_train, y_train)

prediction = model.predict(x_test)

print("Robust TSA Random Forest trained")
print("Training samples:", len(x_train))
print("Testing samples:", len(x_test))

joblib.dump(model, "models/robust_tsa_random_forest.pkl")

print("Model saved successfully")