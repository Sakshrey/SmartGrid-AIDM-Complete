import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import os

data = pd.read_csv("dataset/processed_fdi.csv")

X = data.drop(columns=["attack_type"])
y = data["attack_type"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

prediction = model.predict(X_test)

accuracy = accuracy_score(y_test, prediction)
precision = precision_score(y_test, prediction, pos_label="FDI")
recall = recall_score(y_test, prediction, pos_label="FDI")
f1 = f1_score(y_test, prediction, pos_label="FDI")

print("Random Forest IDS Results")
print("-------------------------")
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, prediction))

os.makedirs("models", exist_ok=True)

joblib.dump(
    model,
    "models/random_forest_ids.pkl",
    compress=3
)

print("\nModel saved successfully")