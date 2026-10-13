import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import joblib
import os

data = pd.read_csv("dataset/processed_fdi.csv")

normal_data = data[data["attack_type"] == "Normal"].copy()

X = normal_data.drop(columns=["attack_type"])

X_train, X_test = train_test_split(
    X,
    test_size=0.2,
    random_state=42
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

input_size = X_train.shape[1]

input_layer = Input(shape=(input_size,))

encoded = Dense(8, activation="relu")(input_layer)
encoded = Dense(4, activation="relu")(encoded)

decoded = Dense(8, activation="relu")(encoded)
decoded = Dense(input_size, activation="linear")(decoded)

autoencoder = Model(input_layer, decoded)

autoencoder.compile(
    optimizer="adam",
    loss="mse"
)

print("Training Autoencoder...")

autoencoder.fit(
    X_train,
    X_train,
    epochs=20,
    batch_size=256,
    validation_data=(X_test, X_test),
    verbose=1
)

print("\nAutoencoder training completed")

train_prediction = autoencoder.predict(X_train, verbose=0)

errors = np.mean(
    np.square(X_train - train_prediction),
    axis=1
)

threshold = np.percentile(errors, 95)

print("Anomaly Threshold:", threshold)

os.makedirs("models", exist_ok=True)

autoencoder.save("models/autoencoder.keras")

joblib.dump(
    scaler,
    "models/autoencoder_scaler.pkl"
)

joblib.dump(
    threshold,
    "models/autoencoder_threshold.pkl"
)

print("\nAutoencoder saved successfully")