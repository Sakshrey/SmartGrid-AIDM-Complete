
import pandas as pd

normal_data = pd.read_csv("dataset/processed_fdi_test.csv")
fdi_data = pd.read_csv("dataset/final_subtle_fdi_test.csv")
tsa_data = pd.read_csv("dataset/final_subtle_tsa_test.csv")

normal = normal_data[
    normal_data["attack_type"] == "Normal"
].copy()

print("Normal dataset columns:")
print(normal.columns.tolist())

print("\nSubtle FDI dataset columns:")
print(fdi_data.columns.tolist())

print("\nSubtle TSA dataset columns:")
print(tsa_data.columns.tolist())

print("\nNormal data summary:")
print(normal.select_dtypes(include="number").describe().T)

print("\nSubtle FDI data summary:")
print(fdi_data.select_dtypes(include="number").describe().T)

print("\nSubtle TSA data summary:")
print(tsa_data.select_dtypes(include="number").describe().T)

print("\nNormal frequency sample:")
print(normal["Actual frequency value"].head(10).tolist())

print("\nSubtle FDI frequency sample:")
print(fdi_data["Actual frequency value"].head(10).tolist())

print("\nSubtle TSA frequency sample:")
print(tsa_data["Actual frequency value"].head(10).tolist())
