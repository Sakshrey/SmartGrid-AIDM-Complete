
import joblib
import pandas as pd

model = joblib.load("models/balanced_unified_random_forest.pkl")

features = [
    "Actual frequency value",
    "Fraction of second",
    "PMU time",
    "PDC time",
    "interarrival time",
    "time difference"
]

importance = model.feature_importances_

result = pd.DataFrame({
    "Feature": features,
    "Importance": importance
})

result = result.sort_values(
    by="Importance",
    ascending=False
)

print("\nRandom Forest Feature Importance")
print("================================")
print(result.to_string(
    index=False,
    formatters={"Importance": "{:.4f}".format}
))

print("\nFeature importance check completed.")
