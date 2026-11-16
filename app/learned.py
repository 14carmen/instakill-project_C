"""The like model trained in week05.ipynb: load like_model.npz once, score signals with `predict`."""
from pathlib import Path

import numpy as np

from app.signals import FEATURES

MODEL_PATH = Path(__file__).parent / "like_model.npz"


def load_model(path: Path = MODEL_PATH) -> dict:
    if not path.exists():
        return {}
    saved = dict(np.load(path, allow_pickle=False))
    if tuple(saved["features"]) != FEATURES:
        raise ValueError(f"{path} expects signals {list(saved['features'])}, InstaKill computes {list(FEATURES)}")
    return saved


saved = load_model()


def is_available() -> bool:
    return bool(saved)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


def predict(features: np.ndarray) -> np.ndarray:
    """Raw signals (n, 6), in the order of FEATURES -> P(like) per row. Same code as the notebook."""
    if not saved:
        raise RuntimeError("like_model.npz is not loaded")
    x = (features - saved["mu"]) / saved["sd"]
    hidden = np.tanh(x @ saved["W1"] + saved["b1"])
    return sigmoid((hidden @ saved["W2"] + saved["b2"])[:, 0])
