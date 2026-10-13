import pandas as pd
import numpy as np

data = pd.read_csv("dataset/processed_tsa.csv")

normal_data = data[data["attack_type"] == "Normal"].copy()

np.random.seed(42)

normal_data["original_frequency"] = normal_data["Actual frequency value"]

change = np.random.uniform(0.001, 0.005, len(normal_data))

normal_data["Actual frequency value"] += change

normal_data["attack_type"] = "TSA_Attack"

normal_data.to_csv("dataset/subtle_tsa_attack.csv", index=False)

print("Subtle TSA attack created")
print("Normal samples:", len(normal_data))
print("Attacked samples:", len(normal_data))

print("Average original frequency:",
      normal_data["original_frequency"].mean())

print("Average attacked frequency:",
      normal_data["Actual frequency value"].mean())