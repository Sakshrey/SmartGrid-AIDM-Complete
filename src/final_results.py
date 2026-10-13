import matplotlib.pyplot as plt
import pandas as pd

models = [
    "Baseline FDI",
    "Robust FDI",
    "AIDM FDI",
    "Baseline TSA",
    "Robust TSA",
    "AIDM TSA"
]

detection = [
    9.18,
    63.29,
    65.26,
    0.00,
    53.11,
    55.43
]

data = pd.DataFrame({
    "Model": models,
    "Detection Rate": detection
})

print("Final Detection Results")
print("-----------------------")
print(data)

data.to_csv(
    "outputs/final_detection_results.csv",
    index=False
)

plt.figure(figsize=(10, 6))

plt.bar(
    data["Model"],
    data["Detection Rate"]
)

plt.ylabel("Detection Rate (%)")
plt.xlabel("Model")
plt.title("Comparison of Intrusion Detection Performance")

plt.xticks(
    rotation=30,
    ha="right"
)

plt.ylim(0, 100)

plt.tight_layout()

plt.savefig(
    "outputs/detection_comparison.png",
    dpi=300
)

plt.show()

print("\nGraph saved successfully")
print("CSV saved successfully")