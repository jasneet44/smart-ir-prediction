import threading

import shap

from app.services.model_loader import (
    get_model,
    VALID_MODEL_TYPES
)


_explainers = {}
_lock = threading.Lock()


def _get_explainer(model_type):
    if model_type not in _explainers:
        with _lock:
            if model_type not in _explainers:
                _explainers[model_type] = shap.TreeExplainer(
                    get_model(model_type)
                )

    return _explainers[model_type]


def get_top_factors(
    model_type,
    features,
    feature_names,
    top_n=5
):

    if model_type not in VALID_MODEL_TYPES:
        raise ValueError("Invalid model_type")

    explainer = _get_explainer(model_type)

    shap_values = explainer(features)

    # RandomForest binary classification handling
    if len(shap_values.values.shape) == 3:
        values = shap_values.values[0, :, 1]
    else:
        values = shap_values.values[0]

    contributions = list(
        zip(feature_names, values)
    )

    contributions = sorted(
        contributions,
        key=lambda x: abs(x[1]),
        reverse=True
    )

    top_factors = []

    for name, val in contributions[:top_n]:

        impact = round(float(val * 100), 3)

        if val > 0:
            text = (
                f"{name} increased risk "
                f"(impact {impact})"
            )
        else:
            text = (
                f"{name} decreased risk "
                f"(impact {impact})"
            )

        top_factors.append(text)

    return top_factors, contributions
