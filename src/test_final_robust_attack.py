import pandas as pd
import joblib

model = joblib.load("models/final_robust_random_forest.pkl")

data = pd.read_csv("dataset/subtle_fdi_attack.csv")

X = data.drop(columns=["attack_type", "original_frequency"])

prediction = model.predict(X)

detected = sum(prediction == "FDI")
missed = sum(prediction == "Normal")

print("Final Robust Model - Subtle FDI Attack")
print("---------------------------------------")

print("Total attacked samples:", len(data))
print("Detected as FDI:", detected)
print("Detected as Normal:", missed)

print("\nDetection Rate:", detected / len(data) * 100, "%")
print("Miss Rate:", missed / len(data) * 100, "%")

print("\nPrediction Distribution:")
print(pd.Series(prediction).value_counts())