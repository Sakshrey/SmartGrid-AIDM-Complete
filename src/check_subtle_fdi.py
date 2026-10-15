import pandas as pd

data = pd.read_csv("dataset/final_subtle_fdi_test.csv")

print("Total rows:", len(data))
print("\nColumns:")
print(data.columns.tolist())

print("\nAttack labels:")
print(data["attack_type"].value_counts())

print("\nOriginal frequency statistics:")
print(data["original_frequency"].describe())

print("\nModified frequency statistics:")
print(data["Actual frequency value"].describe())

difference = (
    data["Actual frequency value"] - data["original_frequency"]
)

print("\nFrequency change statistics:")
print(difference.describe())

print("\nFirst 10 frequency changes:")
print(difference.head(10).to_list())
