import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

data = pd.read_csv("dataset/processed_tsa.csv")

x = data.drop(columns=["attack_type"])
y = data["attack_type"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=30,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)

model.fit(x_train, y_train)

prediction = model.predict(x_test)

accuracy = accuracy_score(y_test, prediction)
precision = precision_score(y_test, prediction, pos_label="TSA")
recall = recall_score(y_test, prediction, pos_label="TSA")
f1 = f1_score(y_test, prediction, pos_label="TSA")

print("TSA Random Forest Results")
print("-------------------------")
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, prediction))

import joblib

joblib.dump(model, "models/tsa_random_forest.pkl")

print("\nModel saved successfully")