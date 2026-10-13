import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("dataset/processed_fdi.csv")

print("Dataset Shape:", data.shape)

print("\nAttack Type Count:")
print(data["attack_type"].value_counts())

print("\nAverage Frequency:")
print(data.groupby("attack_type")["Actual frequency value"].mean())

print("\nAverage Interarrival Time:")
print(data.groupby("attack_type")["interarrival time"].mean())

plt.figure(figsize=(7, 5))
data["attack_type"].value_counts().plot(kind="bar")
plt.title("Normal vs FDI Samples")
plt.xlabel("Attack Type")
plt.ylabel("Number of Samples")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("outputs/attack_distribution.png")
plt.show()

plt.figure(figsize=(8, 5))

normal = data[data["attack_type"] == "Normal"]
fdi = data[data["attack_type"] == "FDI"]

plt.hist(normal["Actual frequency value"], bins=50, alpha=0.6, label="Normal")
plt.hist(fdi["Actual frequency value"], bins=50, alpha=0.6, label="FDI")

plt.title("Frequency Value Distribution")
plt.xlabel("Actual Frequency Value")
plt.ylabel("Number of Samples")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/frequency_distribution.png")
plt.show()

plt.figure(figsize=(8, 5))

plt.hist(normal["interarrival time"], bins=50, alpha=0.6, label="Normal")
plt.hist(fdi["interarrival time"], bins=50, alpha=0.6, label="FDI")

plt.title("Interarrival Time Distribution")
plt.xlabel("Interarrival Time (seconds)")
plt.ylabel("Number of Samples")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/interarrival_distribution.png")
plt.show()

print("\nEDA completed successfully")
print("Graphs saved in outputs folder")