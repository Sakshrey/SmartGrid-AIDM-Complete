import pandas as pd

files = [
    "dataset/processed_fdi.csv",
    "dataset/processed_fdi_test.csv",
    "dataset/final_subtle_fdi_test.csv"
]

for path in files:
    print("\n" + "=" * 50)
    print(path)

    data = pd.read_csv(path)

    print("Rows:", len(data))
    print("Columns:", data.columns.tolist())

    if "attack_type" in data.columns:
        print(data["attack_type"].value_counts())

    if "original_frequency" in data.columns:
        difference = (
            data["Actual frequency value"] -
            data["original_frequency"]
        )
        print("Frequency-change range:",
              difference.min(), "to", difference.max())

print("\nCheck complete.")