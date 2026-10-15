
import pandas as pd

normal_data = pd.read_csv("dataset/processed_fdi_test.csv")
fdi_data = pd.read_csv("dataset/final_subtle_fdi_test.csv")
tsa_data = pd.read_csv("dataset/final_subtle_tsa_test.csv")

normal = normal_data[
    normal_data["attack_type"] == "Normal"
].reset_index(drop=True)

features = [
    "Fraction of second",
    "PMU time",
    "PDC time",
    "interarrival time",
    "time difference"
]

print("Normal rows:", len(normal))
print("Subtle FDI rows:", len(fdi_data))
print("Subtle TSA rows:", len(tsa_data))

print("\nChecking unchanged time features:")

print(
    "Normal and FDI time features match:",
    normal[features].equals(fdi_data[features])
)

print(
    "Normal and TSA time features match:",
    normal[features].equals(tsa_data[features])
)

print("\nChecking original frequency:")

print(
    "FDI original frequency matches Normal:",
    normal["Actual frequency value"].equals(
        fdi_data["original_frequency"]
    )
)

print(
    "TSA original frequency matches Normal:",
    normal["Actual frequency value"].equals(
        tsa_data["original_frequency"]
    )
)

print("\nFrequency perturbation statistics:")

fdi_change = (
    fdi_data["Actual frequency value"]
    - fdi_data["original_frequency"]
)

tsa_change = (
    tsa_data["Actual frequency value"]
    - tsa_data["original_frequency"]
)

print("FDI change:")
print(fdi_change.describe())

print("\nTSA change:")
print(tsa_change.describe())

print("\nFDI and TSA perturbed frequencies match:",
      fdi_data["Actual frequency value"].equals(
          tsa_data["Actual frequency value"]
      ))
