
import pandas as pd

train = pd.read_csv("dataset/processed_fdi.csv")
test = pd.read_csv("dataset/processed_fdi_test.csv")

features = [
    "Actual frequency value",
    "Fraction of second",
    "PMU time",
    "PDC time",
    "interarrival time",
    "time difference"
]

train_normal = train[
    train["attack_type"] == "Normal"
][features].copy()

test_normal = test[
    test["attack_type"] == "Normal"
][features].copy()

train_rows = set(
    map(tuple, train_normal.round(8).to_numpy())
)

test_rows = set(
    map(tuple, test_normal.round(8).to_numpy())
)

overlap = train_rows.intersection(test_rows)

print("Training Normal rows:", len(train_normal))
print("Testing Normal rows:", len(test_normal))
print("Matching unique rows:", len(overlap))
print("Test unique rows:", len(test_rows))

if len(test_rows) > 0:
    print(
        "Test overlap percentage:",
        round(len(overlap) / len(test_rows) * 100, 2),
        "%"
    )
