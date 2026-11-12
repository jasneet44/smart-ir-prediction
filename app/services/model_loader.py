import os
import threading

import joblib

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

MODEL_DIR = os.path.join(
    BASE_DIR,
    "prediction",
    "models"
)

VALID_MODEL_TYPES = ("basic", "intermediate", "advanced")

_models = {}
_lock = threading.Lock()


def get_model(model_type: str):
    """Load a model on first use and cache it."""

    if model_type not in VALID_MODEL_TYPES:
        raise ValueError("Invalid model_type")

    if model_type not in _models:
        with _lock:
            if model_type not in _models:
                _models[model_type] = joblib.load(
                    os.path.join(MODEL_DIR, f"{model_type}.pkl")
                )

    return _models[model_type]
