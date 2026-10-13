import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

model = joblib.load("models/random_forest_ids.pkl")

data = pd.read_csv("dataset/subtle_fdi_attack.csv")

X = data.drop(columns=["attack_type", "original_frequency"])

prediction = model.predict(X)

y = ["FDI"] * len(data)

accuracy = accuracy_score(y, prediction)
precision = precision_score(y, prediction, pos_label="FDI", zero_division=0)
recall = recall_score(y, prediction, pos_label="FDI", zero_division=0)
f1 = f1_score(y, prediction, pos_label="FDI", zero_division=0)

print("Subtle FDI Attack Evaluation")
print("----------------------------")
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)

print("\nConfusion Matrix:")
print(confusion_matrix(y, prediction, labels=["Normal", "FDI"]))

print("\nDetection Rate:", recall * 100, "%")
print("Miss Rate:", (1 - recall) * 100, "%")