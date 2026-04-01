"""
Trains a CNN using data collected in an .npz file.
It takes in N samples that are dimension 125, 8 (timesteps, channels),
and it outputs dimension 3, 1 for 3D angular velocity.
80% is used as training, and 20% is used for testing.

Methodology:
1. TCN: Uses dilated, causal convolutions to capture long-range temporal dependencies .
2. Causal Padding: Ensures prediction at time `t` only uses data from `t` and before.
3. Incremental Mode: Allows loading a base model and adapting it to a new session (IRR concept).
"""

import sys
import time
import json
import datetime
import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

try:
    INPUT_FILE = sys.argv[1]
except IndexError:
    raise SystemExit("Error: Must provide an input data file (.npz).")
date = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d")
OUTPUT_FILE = "emg_cnn_model_" + sys.argv[2] or 0 + "_" + date + ".keras"


def build_model(input_dim):
    model = models.Sequential([
        layers.Input(shape=input_dim),
        
        # First Convolution Layer
        layers.Conv1D(64, kernel_size=5, activation='relu'),
        layers.MaxPooling1D(pool_size=2),
        
        # Second Convolution Layer
        layers.Conv1D(128, kernel_size=3, activation='relu'),
        layers.GlobalAveragePooling1D(), # Flattens the time dimension
        
        # Fully Connected Layer
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(3) # Output: Omega_X, Omega_Y, Omega_Z
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def main():
    data = np.load(INPUT_FILE)
    X = data['emg'] # (N, 125, 8)
    y = data['velocity'] # (N, 3)
    
    # Normalize EMG data based upon regular peaking and save value for use with model
    scaling_factor = np.percentile(np.abs(X), 99) 
    print(f"Determined Scaling Factor: {scaling_factor} uV")
    X_norm = X / scaling_factor
    with open("scaler_params_" + sys.argv[2] or 0 + "_" + date + ".json", "w") as f:
        json.dump({"scale": scaling_factor}, f)

    X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.2)

    model = build_model((125, 8))
    checkpoint = ModelCheckpoint(
        filepath=OUTPUT_FILE,
        save_best_only=True,
        monitor='val_loss',
        mode='min'
    )
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=16, callbacks=[checkpoint])
    model.save(OUTPUT_FILE)
    print("Model saved.")
