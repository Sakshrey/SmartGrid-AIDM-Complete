import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import os

data = pd.read_csv("dataset/processed_fdi.csv")

train_data, test_data = train_test_split(
    data,
    test_size=0.2,
    random_state=42,
    stratify=data["attack_type"]
)

train_normal = train_data[train_data["attack_type"] == "Normal"].copy()

np.random.seed(42)

train_attack = train_normal.copy()

train_attack["Actual frequency value"] = (
    train_attack["Actual frequency value"] +
    np.random.uniform(0.001, 0.005, len(train_attack))
)

train_attack["attack_type"] = "FDI"

train_data["attack_type"] = train_data["attack_type"].replace({
    "Normal": "Normal",
    "FDI": "FDI"
})

combined_train = pd.concat(
    [train_data, train_attack],
    ignore_index=True
)

X_train = combined_train.drop(columns=["attack_type"])
y_train = combined_train["attack_type"]

X_test = test_data.drop(columns=["attack_type"])
y_test = test_data["attack_type"]

print("Training Final Robust Random Forest...")

model = RandomForestClassifier(
    n_estimators=30,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Training completed")

prediction = model.predict(X_test)

accuracy = accuracy_score(y_test, prediction)
precision = precision_score(
    y_test,
    prediction,
    pos_label="FDI",
    zero_division=0
)
recall = recall_score(
    y_test,
    prediction,
    pos_label="FDI",
    zero_division=0
)
f1 = f1_score(
    y_test,
    prediction,
    pos_label="FDI",
    zero_division=0
)

print("\nFinal Robust Model Results")
print("--------------------------")
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)

print("\nConfusion Matrix:")
print(confusion_matrix(
    y_test,
    prediction,
    labels=["Normal", "FDI"]
))

os.makedirs("models", exist_ok=True)

joblib.dump(
    model,
    "models/final_robust_random_forest.pkl",
    compress=3
)

print("\nFinal robust model saved successfully")