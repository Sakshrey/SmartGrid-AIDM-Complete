import pandas as pd

file = "dataset/Clean_FDI_Combined.csv"

data = pd.read_csv(file, low_memory=False)

data = data.drop(columns=[
    "Message Time Quality indicator code",
    "PMU Time Quality",
    "Time synchronized",
    "attack_active",
    "manipulated_fields"
])

def convert_pmu_time(value):
    if pd.isna(value):
        return None

    parts = str(value).split(":")
    return int(parts[0]) * 60 + float(parts[1])

def convert_time(value):
    if pd.isna(value):
        return None

    parts = str(value).split(":")

    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])

    return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])

data["PMU time"] = data["PMU time"].apply(convert_pmu_time)
data["PDC time"] = data["PDC time"].apply(convert_pmu_time)
data["interarrival time"] = data["interarrival time"].apply(convert_time)
data["time difference"] = data["time difference"].apply(convert_time)

data = data.dropna()

data.to_csv("dataset/processed_fdi.csv", index=False)

print("Preprocessing completed")
print("New shape:", data.shape)

print("\nColumns:")
print(data.columns)

print("\nFirst 5 rows:")
print(data.head())