"""
Train a KRR (Kernel Ridge Regression) model
"""

import numpy as np
from sklearn.kernel_ridge import KernelRidge
import joblib

def extract_features(X_raw):
    """Converts (Samples, 125, 8) raw data into (Samples, 8) RMS features"""
    return np.sqrt(np.mean(np.square(X_raw), axis=1))

def train_krr(npz_file, model_name):
    data = np.load(npz_file)
    X_raw, y = data['emg'], data['velocity']
    
    # Data collection already normalizes, so no need to do it here
    X_features = extract_features(X_raw)
    
    model = KernelRidge(alpha=1.0, kernel='rbf', gamma=0.1)
    
    print("Training KRR")
    model.fit(X_features, y)

    joblib.dump(model, f"{model_name}.joblib")
    print(f"KRR Model saved to {model_name}.joblib")
