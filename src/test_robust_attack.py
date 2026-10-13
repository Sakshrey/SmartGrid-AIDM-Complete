import pandas as pd
import joblib

model = joblib.load("models/robust_random_forest.pkl")

data = pd.read_csv("dataset/subtle_fdi_attack.csv")

X = data.drop(columns=["attack_type", "original_frequency"])

prediction = model.predict(X)

print("Robust Model Attack Detection")
print("-----------------------------")

print("Total attacked samples:", len(data))
print("Detected as FDI:", sum(prediction == "FDI"))
print("Detected as Normal:", sum(prediction == "Normal"))

print("\nPrediction Distribution:")
print(pd.Series(prediction).value_counts())