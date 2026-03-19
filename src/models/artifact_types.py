from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import shap
from sklearn.isotonic import IsotonicRegression


@dataclass
class ProbabilityCalibrator:
    method: str
    fitted_on_split: str
    model_name: str
    regressor: IsotonicRegression

    def predict(self, raw_scores: np.ndarray | list[float]) -> np.ndarray:
        arr = np.asarray(raw_scores, dtype=float)
        return np.clip(self.regressor.predict(arr), 0.0, 1.0)


@dataclass
class SerializableShapExplainer:
    model: Any
    background: np.ndarray
    feature_names: list[str]
    _explainer: Any | None = None

    def __call__(self, X: np.ndarray) -> Any:
        if self._explainer is None:
            self._explainer = shap.Explainer(self.model, self.background)
        return self._explainer(X)
