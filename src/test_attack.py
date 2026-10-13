import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

model = joblib.load("models/random_forest_ids.pkl")

data = pd.read_csv("dataset/fdi_attack_data.csv")

X = data.drop(columns=["attack_type", "original_frequency"])
y = data["attack_type"]

prediction = model.predict(X)

print("FDI Attack Detection Results")
print("----------------------------")

print("Total attacked samples:", len(data))
print("Detected as FDI:", sum(prediction == "FDI"))
print("Detected as Normal:", sum(prediction == "Normal"))

print("\nPrediction Distribution:")
print(pd.Series(prediction).value_counts())