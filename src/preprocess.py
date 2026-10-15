import pandas as pd

def convert_pmu_time(value):
    if pd.isna(value):
        return None

    parts = str(value).strip().split(":")

    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])

    return float(value)

def convert_time(value):
    if pd.isna(value):
        return None

    value = str(value).strip()
    parts = value.split(":")

    if len(parts) == 1:
        return float(parts[0])

    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])

    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])

    return None

def preprocess(file, output):
    data = pd.read_csv(file, low_memory=False)

    data = data.drop(columns=[
        "Message Time Quality indicator code",
        "PMU Time Quality",
        "Time synchronized",
        "attack_active",
        "manipulated_fields"
    ], errors="ignore")

    data["PMU time"] = data["PMU time"].apply(convert_pmu_time)
    data["PDC time"] = data["PDC time"].apply(convert_pmu_time)
    data["interarrival time"] = data["interarrival time"].apply(convert_time)

    data["Actual frequency value"] = pd.to_numeric(
        data["Actual frequency value"], errors="coerce"
    )
    data["Fraction of second"] = pd.to_numeric(
        data["Fraction of second"], errors="coerce"
    )

    difference = data["PDC time"] - data["PMU time"]

    # Normalize clock differences to the nearest hour-boundary difference.
    difference = ((difference + 1800) % 3600) - 1800

    data["time difference"] = difference / 60

    data = data.dropna()

    data.to_csv(output, index=False)

    print(output)
    print("Shape:", data.shape)
    print(data["attack_type"].value_counts())
    print("\nTime difference statistics:")
    print(data["time difference"].describe())
    print()

preprocess(
    "dataset/Clean_FDI_Combined.csv",
    "dataset/processed_fdi.csv"
)

print("Training preprocessing completed.")