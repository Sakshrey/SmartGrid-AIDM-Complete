
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

model = joblib.load("models/balanced_unified_random_forest.pkl")

fdi = pd.read_csv("dataset/processed_fdi_test.csv")
tsa = pd.read_csv("dataset/processed_tsa_test.csv")
subtle_fdi = pd.read_csv("dataset/final_subtle_fdi_test.csv")
subtle_tsa = pd.read_csv("dataset/final_subtle_tsa_test.csv")

normal = fdi[fdi["attack_type"] == "Normal"].sample(10000, random_state=42).copy()
original_fdi = fdi[fdi["attack_type"] == "FDI"].sample(10000, random_state=42).copy()
original_tsa = tsa[tsa["attack_type"] == "TSA"].sample(10000, random_state=42).copy()
subtle_fdi = subtle_fdi.sample(10000, random_state=42).copy()
subtle_tsa = subtle_tsa.sample(10000, random_state=42).copy()

normal["true_label"] = "Normal"
original_fdi["true_label"] = "FDI"
original_tsa["true_label"] = "TSA"
subtle_fdi["true_label"] = "FDI"
subtle_tsa["true_label"] = "TSA"

data = pd.concat(
    [normal, original_fdi, subtle_fdi, original_tsa, subtle_tsa],
    ignore_index=True
)

x = data.drop(
    columns=["attack_type", "true_label", "original_frequency"],
    errors="ignore"
)
y = data["true_label"]

prediction = model.predict(x)
labels = ["Normal", "FDI", "TSA"]
cm = confusion_matrix(y, prediction, labels=labels)

print("\nRandom Forest Only Evaluation")
print("-----------------------------")
print("Accuracy:", round(accuracy_score(y, prediction) * 100, 2), "%")
print("\nConfusion Matrix:")
print(cm)
print("\nClassification Report:")
print(classification_report(y, prediction, labels=labels, zero_division=0))

normal_total = sum(cm[0])
false_alarms = normal_total - cm[0][0]
print("False Alarm Rate:", round(false_alarms / normal_total * 100, 2), "%")
print("FDI Detection Rate:", round(cm[1][1] / sum(cm[1]) * 100, 2), "%")
print("TSA Detection Rate:", round(cm[2][2] / sum(cm[2]) * 100, 2), "%")
