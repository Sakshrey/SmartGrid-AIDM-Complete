import pandas as pd
import numpy as np


def convert_pmu_time(value):
    if pd.isna(value):
        return np.nan

    parts = str(value).strip().split(":")

    try:
        if len(parts) == 2:
            return float(parts[0]) * 60 + float(parts[1])

        if len(parts) == 3:
            return (
                float(parts[0]) * 3600
                + float(parts[1]) * 60
                + float(parts[2])
            )

        return float(value)

    except (ValueError, TypeError):
        return np.nan


def convert_time(value):
    if pd.isna(value):
        return np.nan

    value = str(value).strip()
    parts = value.split(":")

    try:
        if len(parts) == 1:
            return float(parts[0])

        if len(parts) == 2:
            return float(parts[0]) * 60 + float(parts[1])

        if len(parts) == 3:
            return (
                float(parts[0]) * 3600
                + float(parts[1]) * 60
                + float(parts[2])
            )

        return np.nan

    except (ValueError, TypeError):
        return np.nan


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

    data["interarrival time"] = data["interarrival time"].apply(
        convert_time
    )

    data["Actual frequency value"] = pd.to_numeric(
        data["Actual frequency value"],
        errors="coerce"
    )

    data["Fraction of second"] = pd.to_numeric(
        data["Fraction of second"],
        errors="coerce"
    )

    # Correct for timestamps wrapping around the 3600-second boundary.
    difference = data["PDC time"] - data["PMU time"]
    difference = ((difference + 1800) % 3600) - 1800

    data["time difference"] = difference / 60

    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna().reset_index(drop=True)

    data.to_csv(output, index=False)

    print("\n", output)
    print("Shape:", data.shape)
    print("\nClass distribution:")
    print(data["attack_type"].value_counts())

    print("\nTime difference statistics:")
    print(data["time difference"].describe())


preprocess(
    "dataset/Clean_FDI_Test.csv",
    "dataset/processed_fdi_test.csv"
)

preprocess(
    "dataset/Clean_TSA_Test.csv",
    "dataset/processed_tsa_test.csv"
)

print("\nTest preprocessing completed.")