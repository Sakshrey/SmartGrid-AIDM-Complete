import pandas as pd
import numpy as np

data = pd.read_csv("dataset/processed_fdi.csv")

normal_data = data[data["attack_type"] == "Normal"].copy()

print("Normal samples:", len(normal_data))

np.random.seed(42)

normal_data["original_frequency"] = normal_data["Actual frequency value"]

change = np.random.uniform(0.05, 0.15, len(normal_data))

normal_data["Actual frequency value"] = (
    normal_data["Actual frequency value"] + change
)

normal_data["attack_type"] = "FDI_Attack"

normal_data.to_csv("dataset/fdi_attack_data.csv", index=False)

print("\nFDI attack simulation completed")
print("Attacked samples:", len(normal_data))
print("Average original frequency:",
      normal_data["original_frequency"].mean())
print("Average attacked frequency:",
      normal_data["Actual frequency value"].mean())

print("\nSaved as:")
print("dataset/fdi_attack_data.csv")