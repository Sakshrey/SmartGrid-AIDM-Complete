import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import os

normal_data = pd.read_csv("dataset/processed_fdi.csv")
attack_data = pd.read_csv("dataset/subtle_fdi_attack.csv")

normal_data = normal_data.drop(columns=["original_frequency"], errors="ignore")
attack_data = attack_data.drop(columns=["original_frequency"], errors="ignore")

attack_data["attack_type"] = "FDI"

data = pd.concat([normal_data, attack_data], ignore_index=True)

X = data.drop(columns=["attack_type"])
y = data["attack_type"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training Robust Random Forest...")

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

print("\nRobust Random Forest Results")
print("----------------------------")
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, prediction, labels=["Normal", "FDI"]))

os.makedirs("models", exist_ok=True)

joblib.dump(
    model,
    "models/robust_random_forest.pkl",
    compress=3
)

print("\nRobust model saved successfully")