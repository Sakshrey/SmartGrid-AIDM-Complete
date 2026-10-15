import pandas as pd
import numpy as np

np.random.seed(42)

print("Loading test datasets...")

fdi = pd.read_csv("dataset/processed_fdi_test.csv")
tsa = pd.read_csv("dataset/processed_tsa_test.csv")

fdi_normal = fdi[fdi["attack_type"] == "Normal"].copy()
tsa_normal = tsa[tsa["attack_type"] == "Normal"].copy()

print("Normal FDI dataset samples:", len(fdi_normal))
print("Normal TSA dataset samples:", len(tsa_normal))

if fdi_normal.empty or tsa_normal.empty:
    raise ValueError("Normal samples are missing from the test datasets.")

# --------------------------------------------------
# Generate subtle FDI attacks
# --------------------------------------------------

print("\nGenerating subtle FDI samples...")

subtle_fdi = fdi_normal.copy()

subtle_fdi["original_frequency"] = (
    subtle_fdi["Actual frequency value"].copy()
)

frequency_change = np.random.uniform(
    0.001, 0.005, len(subtle_fdi)
)

subtle_fdi["Actual frequency value"] += frequency_change
subtle_fdi["attack_type"] = "FDI"

# --------------------------------------------------
# Generate synthetic TSA attacks
# --------------------------------------------------

print("Generating synthetic TSA samples...")

subtle_tsa = tsa_normal.copy()

subtle_tsa["original_frequency"] = (
    subtle_tsa["Actual frequency value"].copy()
)

# Shift PMU time by five minutes.
# The timestamp uses a 0–3599 second cycle.
subtle_tsa["PMU time"] = (
    subtle_tsa["PMU time"] + 300
) % 3600

# Calculate time difference in minutes.
# Adjust for the timestamp cycle boundary.
time_difference = (
    subtle_tsa["PDC time"] - subtle_tsa["PMU time"]
) / 60

time_difference = ((time_difference + 30) % 60) - 30

subtle_tsa["time difference"] = time_difference
subtle_tsa["attack_type"] = "TSA"

# --------------------------------------------------
# Save generated datasets
# --------------------------------------------------

subtle_fdi.to_csv(
    "dataset/final_subtle_fdi_test.csv",
    index=False
)

subtle_tsa.to_csv(
    "dataset/final_subtle_tsa_test.csv",
    index=False
)

# --------------------------------------------------
# Display results
# --------------------------------------------------

print("\nFinal test attacks created")

print("Subtle FDI test samples:", len(subtle_fdi))
print("Subtle TSA test samples:", len(subtle_tsa))

print("\nFDI frequency comparison:")
print("Original average:", subtle_fdi["original_frequency"].mean())
print("Attacked average:", subtle_fdi["Actual frequency value"].mean())

print("\nTSA frequency comparison:")
print("Original average:", subtle_tsa["original_frequency"].mean())
print("Attacked average:", subtle_tsa["Actual frequency value"].mean())

print("\nTSA time difference after shift:")
print(subtle_tsa["time difference"].describe())

print("\nTSA PMU time after shift:")
print(subtle_tsa["PMU time"].describe())

print(
    "\nRows with time difference below -10 minutes:",
    (subtle_tsa["time difference"] < -10).sum()
)

print("\nSynthetic attack files saved successfully.")