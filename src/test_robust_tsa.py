import pandas as pd
import joblib

data = pd.read_csv("dataset/subtle_tsa_attack.csv")

model = joblib.load("models/robust_tsa_random_forest.pkl")

x = data.drop(columns=[
    "attack_type",
    "original_frequency"
])

prediction = model.predict(x)

detected = sum(prediction == "TSA")
normal = sum(prediction == "Normal")

print("Total attacked samples:", len(data))
print("Detected as TSA:", detected)
print("Detected as Normal:", normal)

detection_rate = detected / len(data) * 100
miss_rate = normal / len(data) * 100

print("Detection Rate:", detection_rate, "%")
print("Miss Rate:", miss_rate, "%")