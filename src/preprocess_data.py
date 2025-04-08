import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

def preprocess_for_model(csv_path="data/final_dataset.csv", seq_len=10):
    df = pd.read_csv(csv_path)
    df = df.drop(columns=["Date"])  # We'll drop Date for training

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)

    X, y = [], []
    for i in range(len(scaled) - seq_len):
        X.append(scaled[i:i + seq_len])         # Sequence of length `seq_len`
        y.append(scaled[i + seq_len][0])        # Predicting next Close price

    X, y = np.array(X), np.array(y)
    print(f"✅ Created sequences: X shape = {X.shape}, y shape = {y.shape}")

    # Train-test split (80% train)
    split = int(0.8 * len(X))
    return X[:split], X[split:], y[:split], y[split:], scaler

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler = preprocess_for_model()
