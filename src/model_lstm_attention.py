import numpy as np
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Attention, Concatenate
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt
from preprocess_data import preprocess_for_model

def build_model(input_shape):
    inp = Input(shape=input_shape)
    lstm_out = LSTM(64, return_sequences=True)(inp)

    # Attention layer
    attention = Attention()([lstm_out, lstm_out])
    context = Concatenate()([lstm_out, attention])
    
    # Reduce dimensions
    context = LSTM(32)(context)
    dense = Dense(16, activation='relu')(context)
    output = Dense(1)(dense)

    model = Model(inputs=inp, outputs=output)
    model.compile(optimizer=Adam(0.001), loss='mse')
    return model

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler = preprocess_for_model()

    model = build_model(X_train.shape[1:])
    model.summary()

    print("\n📈 Training model...")
    history = model.fit(X_train, y_train, epochs=100, batch_size=16, 
                       validation_split=0.2, verbose=1)

    print("\n✅ Training complete.")

    # Predict and inverse scale
    preds = model.predict(X_test).flatten()
    y_test_true = y_test

    # Evaluate
    mse = mean_squared_error(y_test_true, preds)
    rmse = sqrt(mse)
    print(f"\n📊 Test Set Metrics:\nMSE: {mse:.4f}\nRMSE: {rmse:.4f}")

    # Plot training history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)

    # Plot predictions
    plt.subplot(1, 2, 2)
    plt.plot(y_test_true, label="Actual", alpha=0.8)
    plt.plot(preds, label="Predicted", alpha=0.8)
    plt.title("Predicted vs Actual")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("outputs/lstm_attention_results.png")
    print("\n📊 Plots saved to outputs/lstm_attention_results.png")
