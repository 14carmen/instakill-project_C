import numpy as np
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "like_model.npz"

if MODEL_PATH.exists():
    saved = dict(np.load(MODEL_PATH))
else:
    saved = {}

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))

def predict(features: np.ndarray) -> np.ndarray:
    if not saved:
        return np.zeros(len(features))
    x = (features - saved["mu"]) / saved["sd"]
    hidden = np.tanh(x @ saved["W1"] + saved["b1"])
    return sigmoid((hidden @ saved["W2"] + saved["b2"])[:, 0])