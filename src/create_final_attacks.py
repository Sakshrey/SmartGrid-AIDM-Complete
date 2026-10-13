import pandas as pd
import numpy as np

np.random.seed(42)

fdi = pd.read_csv("dataset/processed_fdi_test.csv")
tsa = pd.read_csv("dataset/processed_tsa_test.csv")

fdi_normal = fdi[fdi["attack_type"] == "Normal"].copy()
tsa_normal = tsa[tsa["attack_type"] == "Normal"].copy()

fdi_normal["original_frequency"] = fdi_normal["Actual frequency value"]

fdi_normal["Actual frequency value"] += np.random.uniform(
    0.001,
    0.005,
    len(fdi_normal)
)

fdi_normal["attack_type"] = "FDI"

tsa_normal["original_frequency"] = tsa_normal["Actual frequency value"]

tsa_normal["Actual frequency value"] += np.random.uniform(
    0.001,
    0.005,
    len(tsa_normal)
)

tsa_normal["attack_type"] = "TSA"

fdi_normal.to_csv(
    "dataset/final_subtle_fdi_test.csv",
    index=False
)

tsa_normal.to_csv(
    "dataset/final_subtle_tsa_test.csv",
    index=False
)

print("Final test attacks created")

print("\nSubtle FDI test samples:", len(fdi_normal))
print("Subtle TSA test samples:", len(tsa_normal))

print("\nAverage FDI original frequency:",
      fdi_normal["original_frequency"].mean())

print("Average FDI attacked frequency:",
      fdi_normal["Actual frequency value"].mean())

print("\nAverage TSA original frequency:",
      tsa_normal["original_frequency"].mean())

print("Average TSA attacked frequency:",
      tsa_normal["Actual frequency value"].mean())