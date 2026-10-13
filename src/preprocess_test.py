import pandas as pd

def convert_pmu_time(value):
    if pd.isna(value):
        return None

    parts = str(value).split(":")

    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])

    return float(value)

def convert_time(value):
    if pd.isna(value):
        return None

    value = str(value)
    parts = value.split(":")

    if len(parts) == 1:
        return float(parts[0])

    if len(parts) == 2:
        return float(parts[0]) * 60 + float(parts[1])

    if len(parts) == 3:
        return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])

    return None

def preprocess(file, output):
    data = pd.read_csv(file, low_memory=False)

    data = data.drop(columns=[
        "Message Time Quality indicator code",
        "PMU Time Quality",
        "Time synchronized",
        "attack_active",
        "manipulated_fields"
    ])

    data["PMU time"] = data["PMU time"].apply(convert_pmu_time)
    data["PDC time"] = data["PDC time"].apply(convert_pmu_time)
    data["interarrival time"] = data["interarrival time"].apply(convert_time)
    data["time difference"] = data["time difference"].apply(convert_time)

    data = data.dropna()

    data.to_csv(output, index=False)

    print(output)
    print("Shape:", data.shape)
    print(data["attack_type"].value_counts())
    print()

preprocess(
    "dataset/Clean_FDI_Test.csv",
    "dataset/processed_fdi_test.csv"
)

preprocess(
    "dataset/Clean_TSA_Test.csv",
    "dataset/processed_tsa_test.csv"
)

print("Test preprocessing completed")