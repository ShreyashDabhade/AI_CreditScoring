"""Stable probability calibration helpers for persisted model artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def _as_score_vector(raw_pd: Any) -> np.ndarray:
    return np.asarray(raw_pd, dtype=float).reshape(-1)


@dataclass
class SigmoidCalibrator:
    estimator: LogisticRegression
    method: str = "sigmoid"

    def predict(self, raw_pd: Any) -> np.ndarray:
        scores = _as_score_vector(raw_pd).reshape(-1, 1)
        return self.estimator.predict_proba(scores)[:, 1]


@dataclass
class IsotonicCalibrator:
    estimator: IsotonicRegression
    method: str = "isotonic"

    def predict(self, raw_pd: Any) -> np.ndarray:
        scores = _as_score_vector(raw_pd)
        return np.clip(self.estimator.predict(scores), 0.0, 1.0)


ProbabilityCalibrator = SigmoidCalibrator | IsotonicCalibrator


def fit_probability_calibrator(
    raw_pd: Any,
    y_true: Any,
    *,
    method: str = "sigmoid",
    random_state: int | None = None,
) -> ProbabilityCalibrator:
    scores = _as_score_vector(raw_pd)
    labels = np.asarray(y_true, dtype=int).reshape(-1)
    method_name = method.strip().lower()

    if method_name == "sigmoid":
        estimator = LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            random_state=random_state,
        )
        estimator.fit(scores.reshape(-1, 1), labels)
        return SigmoidCalibrator(estimator=estimator)

    if method_name == "isotonic":
        estimator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        estimator.fit(scores, labels.astype(float))
        return IsotonicCalibrator(estimator=estimator)

    raise ValueError(
        f"Unsupported calibration method: {method}. Expected 'sigmoid' or 'isotonic'."
    )
