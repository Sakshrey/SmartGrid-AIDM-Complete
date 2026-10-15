
import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, classification_report

model = joblib.load("models/balanced_unified_random_forest.pkl")

fdi = pd.read_csv("dataset/processed_fdi_test.csv")

normal = fdi[fdi["attack_type"] == "Normal"].sample(
    10000, random_state=42
)

x = normal.drop(columns=["attack_type"], errors="ignore")

print("Testing Random Forest on normal samples...")
prediction = model.predict(x)

labels = ["Normal", "FDI", "TSA"]

print("\nPrediction Distribution")
print(pd.Series(prediction).value_counts())

print("\nNormal Sample Results")
print("Total normal samples:", len(normal))
print("Correctly classified as Normal:",
      sum(prediction == "Normal"))
print("Incorrectly flagged as attack:",
      sum(prediction != "Normal"))
print("False Alarm Rate:",
      round(sum(prediction != "Normal") / len(normal) * 100, 2), "%")

print("\nConfusion Matrix")
print(confusion_matrix(
    ["Normal"] * len(prediction),
    prediction,
    labels=labels
))
