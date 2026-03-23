"""Runtime helpers for persisted Module 3 artifacts."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression


class TreeShapExplainer:
    """Pickle-friendly SHAP wrapper around a tree model."""

    def __init__(self, model: Any):
        self.model = model
        self._explainer = None

    def _get_explainer(self):
        if self._explainer is None:
            import shap

            self._explainer = shap.TreeExplainer(self.model)
        return self._explainer

    def __call__(self, X: Any):
        arr = np.asarray(X, dtype=float)
        return self._get_explainer()(arr)

    def shap_values(self, X: Any):
        arr = np.asarray(X, dtype=float)
        return self._get_explainer().shap_values(arr)

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_explainer"] = None
        return state


class LogisticProbabilityCalibrator:
    """Pickle-friendly probability calibrator with a stable predict API."""

    def __init__(self):
        self.model = LogisticRegression(max_iter=1000)

    def fit(self, raw_pd: Any, y_true: Any) -> "LogisticProbabilityCalibrator":
        x = np.asarray(raw_pd, dtype=float).reshape(-1, 1)
        y = np.asarray(y_true, dtype=int).reshape(-1)
        self.model.fit(x, y)
        return self

    def predict(self, raw_pd: Any) -> np.ndarray:
        x = np.asarray(raw_pd, dtype=float).reshape(-1, 1)
        return self.model.predict_proba(x)[:, 1]
