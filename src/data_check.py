import pandas as pd

file = "dataset/Clean_FDI_Combined.csv"

data = pd.read_csv(file, low_memory=False)

print("Shape:")
print(data.shape)

print("\nAttack types:")
print(data["attack_type"].value_counts())

print("\nUnique values in each column:")

for col in data.columns:
    print("\n", col)
    print("Unique:", data[col].nunique())
    print(data[col].dropna().unique()[:10])