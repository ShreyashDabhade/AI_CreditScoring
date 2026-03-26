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


class WeightedBlendModel:
    """Pickle-friendly weighted blend over FULL XGBoost and LightGBM models."""

    def __init__(
        self,
        xgboost_model: Any,
        lightgbm_model: Any,
        *,
        weight_xgboost: float,
        weight_lightgbm: float,
        feature_count: int,
    ):
        total = float(weight_xgboost + weight_lightgbm)
        if total <= 0.0:
            raise ValueError("Blend weights must sum to a positive value")
        self.xgboost_model = xgboost_model
        self.lightgbm_model = lightgbm_model
        self.weight_xgboost = float(weight_xgboost / total)
        self.weight_lightgbm = float(weight_lightgbm / total)
        self.n_features_in_ = int(feature_count)

    def _coerce_inputs(self, X: Any) -> tuple[Any, np.ndarray]:
        if hasattr(X, "to_numpy"):
            matrix = X.to_numpy(dtype=float, copy=False)
            xgboost_input = X
        else:
            matrix = np.asarray(X, dtype=float)
            xgboost_input = matrix
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
            xgboost_input = matrix
        return xgboost_input, matrix

    def predict_raw_pd(self, X: Any) -> np.ndarray:
        xgboost_input, lightgbm_input = self._coerce_inputs(X)
        xgb_pd = np.asarray(self.xgboost_model.predict_proba(xgboost_input), dtype=float)[:, 1]
        lgbm_pd = np.asarray(self.lightgbm_model.predict_proba(lightgbm_input), dtype=float)[:, 1]
        blended = self.weight_xgboost * xgb_pd + self.weight_lightgbm * lgbm_pd
        return np.clip(blended, 0.0, 1.0)

    def predict_proba(self, X: Any) -> np.ndarray:
        raw_pd = self.predict_raw_pd(X)
        return np.column_stack([1.0 - raw_pd, raw_pd])


class WeightedBlendShapExplainer:
    """Approximate blend explanations via weighted component Tree SHAP values."""

    def __init__(
        self,
        xgboost_explainer: Any,
        lightgbm_explainer: Any,
        *,
        weight_xgboost: float,
        weight_lightgbm: float,
        explanation_policy: str = "weighted_component_tree_shap_average",
    ):
        total = float(weight_xgboost + weight_lightgbm)
        if total <= 0.0:
            raise ValueError("Blend weights must sum to a positive value")
        self.xgboost_explainer = xgboost_explainer
        self.lightgbm_explainer = lightgbm_explainer
        self.weight_xgboost = float(weight_xgboost / total)
        self.weight_lightgbm = float(weight_lightgbm / total)
        self.explanation_policy = explanation_policy

    def __call__(self, X: Any) -> np.ndarray:
        arr = np.asarray(X, dtype=float)
        xgb_values = _normalize_shap_output(self.xgboost_explainer(arr))
        lgbm_values = _normalize_shap_output(self.lightgbm_explainer(arr))
        return self.weight_xgboost * xgb_values + self.weight_lightgbm * lgbm_values

    def shap_values(self, X: Any) -> np.ndarray:
        return self(X)


def _normalize_shap_output(raw_shap: Any) -> np.ndarray:
    try:
        import shap  # type: ignore
    except Exception:
        shap = None

    if shap is not None and isinstance(raw_shap, shap.Explanation):
        raw_shap = raw_shap.values

    if isinstance(raw_shap, list):
        if not raw_shap:
            raise RuntimeError("Explainer returned an empty SHAP list")
        raw_shap = raw_shap[1] if len(raw_shap) > 1 else raw_shap[0]

    values = np.asarray(raw_shap, dtype=float)
    if values.ndim == 3:
        class_index = 1 if values.shape[-1] > 1 else 0
        values = values[..., class_index]
    return values
