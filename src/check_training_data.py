
import pandas as pd
import joblib

print("Loading model and datasets...")

model = joblib.load("models/balanced_unified_random_forest.pkl")

fdi = pd.read_csv("dataset/processed_fdi.csv")
tsa = pd.read_csv("dataset/processed_tsa.csv")
fdi_test = pd.read_csv("dataset/processed_fdi_test.csv")
tsa_test = pd.read_csv("dataset/processed_tsa_test.csv")

print("\n1. MODEL INFORMATION")
print("--------------------")
print("Model classes:", model.classes_)

if hasattr(model, "feature_names_in_"):
    print("Model training features:")
    print(list(model.feature_names_in_))

print("\n2. DATASET COLUMNS")
print("-----------------")
print("FDI training:", list(fdi.columns))
print("TSA training:", list(tsa.columns))
print("FDI testing:", list(fdi_test.columns))
print("TSA testing:", list(tsa_test.columns))

print("\n3. TRAINING CLASS DISTRIBUTION")
print("------------------------------")
print("FDI dataset:")
print(fdi["attack_type"].value_counts())

print("\nTSA dataset:")
print(tsa["attack_type"].value_counts())

print("\n4. TEST CLASS DISTRIBUTION")
print("--------------------------")
print("FDI test dataset:")
print(fdi_test["attack_type"].value_counts())

print("\nTSA test dataset:")
print(tsa_test["attack_type"].value_counts())

print("\n5. FEATURE COMPATIBILITY")
print("------------------------")

if hasattr(model, "feature_names_in_"):
    expected = list(model.feature_names_in_)
    actual = [
        col for col in fdi_test.columns
        if col not in ["attack_type", "original_frequency", "true_label"]
    ]

    print("Expected feature count:", len(expected))
    print("Test feature count:", len(actual))
    print("Missing model features:", [c for c in expected if c not in actual])
    print("Extra test features:", [c for c in actual if c not in expected])

print("\n6. NORMAL FREQUENCY SUMMARY")
print("---------------------------")

for name, data in [
    ("FDI training normal", fdi[fdi["attack_type"] == "Normal"]),
    ("FDI test normal", fdi_test[fdi_test["attack_type"] == "Normal"])
]:
    if "Actual frequency value" in data.columns:
        values = data["Actual frequency value"]
        print(name)
        print("Count:", len(values))
        print("Mean:", round(values.mean(), 6))
        print("Std:", round(values.std(), 6))
        print("Min:", round(values.min(), 6))
        print("Max:", round(values.max(), 6))

print("\nData inspection completed.")
