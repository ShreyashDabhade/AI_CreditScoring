"""Module 3 - XGBoost training, calibration, and artifact serialization."""

from __future__ import annotations

import json
import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from xgboost import XGBClassifier

if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

PROJECT_ROOT = Path(__file__).resolve().parents[2]

from configs.config import (
    APPROVE_THRESHOLD,
    ARTIFACT_DIR,
    DATA_DIR,
    DECLINE_THRESHOLD,
    MODEL_VERSIONS,
    RANDOM_STATE,
)
from src.feature_engineering import (
    build_full,
    build_reduced,
    fit_full_builder,
    fit_reduced_builder,
)
from src.models.calibration import ProbabilityCalibrator, fit_probability_calibrator

FULL_MODEL_PATH = "full_model.joblib"
REDUCED_MODEL_PATH = "reduced_model.joblib"
FULL_CALIBRATOR_PATH = "full_calibrator.joblib"
REDUCED_CALIBRATOR_PATH = "reduced_calibrator.joblib"
FULL_EXPLAINER_PATH = "full_shap_explainer.joblib"
REDUCED_EXPLAINER_PATH = "reduced_shap_explainer.joblib"
REPRODUCIBILITY_REPORT_PATH = "reproducibility_report.json"
RAW_FULL_TABLES = (
    "bureau.csv",
    "previous_application.csv",
    "installments_payments.csv",
    "POS_CASH_balance.csv",
    "credit_card_balance.csv",
)
DEFAULT_CALIBRATION_METHOD = "sigmoid"
DEFAULT_CLASSIFICATION_THRESHOLD = 0.5
XGB_TRAINING_PROFILE = {
    "n_estimators": 800,
    "learning_rate": 0.03,
    "max_depth": 4,
    "min_child_weight": 8,
    "subsample": 0.75,
    "colsample_bytree": 0.75,
    "reg_lambda": 2.0,
    "reg_alpha": 0.1,
    "gamma": 0.1,
}


@dataclass(frozen=True)
class TierArtifacts:
    tier: str
    builder: Any
    model: XGBClassifier
    calibrator: ProbabilityCalibrator
    explainer: Any
    metrics: dict[str, dict[str, float]]
    feature_count: int
    calibration_method: str


def decision_from_pd(probability_of_default: float) -> str:
    if probability_of_default < APPROVE_THRESHOLD:
        return "APPROVE"
    if probability_of_default < DECLINE_THRESHOLD:
        return "REVIEW"
    return "DECLINE"


def _load_pickle_df(path: str) -> pd.DataFrame:
    with open(path, "rb") as handle:
        df = pickle.load(handle)
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{path} did not contain a pandas DataFrame")
    return df


def _resolve_project_path(path: str) -> str:
    candidate = Path(path)
    if candidate.is_absolute():
        return str(candidate)
    return str((PROJECT_ROOT / candidate).resolve())


def _load_processed_splits(processed_dir: str) -> dict[str, pd.DataFrame]:
    resolved_dir = _resolve_project_path(processed_dir)
    return {
        "train": _load_pickle_df(os.path.join(resolved_dir, "train.pkl")),
        "val_model": _load_pickle_df(os.path.join(resolved_dir, "val_model.pkl")),
        "val_policy": _load_pickle_df(os.path.join(resolved_dir, "val_policy.pkl")),
        "test": _load_pickle_df(os.path.join(resolved_dir, "test.pkl")),
    }


def _has_full_raw_tables(raw_dir: str) -> bool:
    resolved_dir = _resolve_project_path(raw_dir)
    return all(os.path.exists(os.path.join(resolved_dir, name)) for name in RAW_FULL_TABLES)


def _fit_xgb_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> XGBClassifier:
    positives = float(y_train.sum())
    negatives = float(len(y_train) - positives)
    scale_pos_weight = negatives / positives if positives > 0 else 1.0

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        random_state=RANDOM_STATE,
        tree_method="hist",
        scale_pos_weight=scale_pos_weight,
        early_stopping_rounds=50,
        **XGB_TRAINING_PROFILE,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    return model


def _fit_calibrator(
    raw_pd: np.ndarray,
    y_true: pd.Series,
    *,
    method: str = DEFAULT_CALIBRATION_METHOD,
) -> ProbabilityCalibrator:
    return fit_probability_calibrator(
        raw_pd,
        y_true,
        method=method,
        random_state=RANDOM_STATE,
    )


def _metrics(
    y_true: pd.Series,
    probs: np.ndarray,
    *,
    threshold: float = DEFAULT_CLASSIFICATION_THRESHOLD,
) -> dict[str, float]:
    y_true_arr = np.asarray(y_true, dtype=int).reshape(-1)
    probs_arr = np.asarray(probs, dtype=float).reshape(-1)
    probs_clipped = np.clip(probs_arr, 1e-6, 1.0 - 1e-6)
    predictions = (probs_arr >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true_arr, probs_arr)),
        "pr_auc": float(average_precision_score(y_true_arr, probs_arr)),
        "brier": float(brier_score_loss(y_true_arr, probs_arr)),
        "logloss": float(log_loss(y_true_arr, probs_clipped, labels=[0, 1])),
        "accuracy": float((predictions == y_true_arr).mean()),
    }


def _evaluate_model(
    model: XGBClassifier,
    calibrator: ProbabilityCalibrator,
    matrices: dict[str, tuple[pd.DataFrame, pd.Series]],
) -> dict[str, dict[str, float]]:
    output: dict[str, dict[str, float]] = {}
    for split_name, (X_split, y_split) in matrices.items():
        raw_pd = model.predict_proba(X_split)[:, 1]
        cal_pd = calibrator.predict(raw_pd)
        raw_metrics = _metrics(y_split, raw_pd)
        cal_metrics = _metrics(y_split, cal_pd)
        output[split_name] = {
            "raw_roc_auc": raw_metrics["roc_auc"],
            "raw_pr_auc": raw_metrics["pr_auc"],
            "raw_accuracy": raw_metrics["accuracy"],
            "calibrated_roc_auc": cal_metrics["roc_auc"],
            "calibrated_pr_auc": cal_metrics["pr_auc"],
            "calibrated_brier": cal_metrics["brier"],
            "calibrated_logloss": cal_metrics["logloss"],
            "calibrated_accuracy": cal_metrics["accuracy"],
        }
    return output


def _save_joblib(obj: Any, artifact_dir: str, filename: str) -> str:
    resolved_dir = _resolve_project_path(artifact_dir)
    os.makedirs(resolved_dir, exist_ok=True)
    path = os.path.join(resolved_dir, filename)
    joblib.dump(obj, path)
    return path


def _build_tier_artifacts(
    *,
    tier: str,
    splits: dict[str, pd.DataFrame],
    build_fn: Callable[..., pd.DataFrame],
    fit_builder_fn: Callable[..., Any],
    raw_dir: str | None = None,
    calibration_method: str = DEFAULT_CALIBRATION_METHOD,
) -> TierArtifacts:
    resolved_raw_dir = _resolve_project_path(raw_dir) if raw_dir is not None else None
    if tier.upper() == "FULL":
        if resolved_raw_dir is None or not _has_full_raw_tables(resolved_raw_dir):
            raise FileNotFoundError(
                "FULL training requires data/raw/ with bureau, previous_application, "
                "installments_payments, POS_CASH_balance, and credit_card_balance tables"
            )
        builder = fit_builder_fn(splits["train"], raw_dir=resolved_raw_dir)
        X_train = build_fn(splits["train"], builder, raw_dir=resolved_raw_dir)
        X_val_model = build_fn(splits["val_model"], builder, raw_dir=resolved_raw_dir)
        X_val_policy = build_fn(splits["val_policy"], builder, raw_dir=resolved_raw_dir)
        X_test = build_fn(splits["test"], builder, raw_dir=resolved_raw_dir)
    else:
        builder = fit_builder_fn(splits["train"])
        X_train = build_fn(splits["train"], builder)
        X_val_model = build_fn(splits["val_model"], builder)
        X_val_policy = build_fn(splits["val_policy"], builder)
        X_test = build_fn(splits["test"], builder)

    y_train = splits["train"]["TARGET"].astype(int)
    y_val_model = splits["val_model"]["TARGET"].astype(int)
    y_val_policy = splits["val_policy"]["TARGET"].astype(int)
    y_test = splits["test"]["TARGET"].astype(int)

    model = _fit_xgb_classifier(X_train, y_train, X_val_model, y_val_model)
    val_model_raw = model.predict_proba(X_val_model)[:, 1]
    calibrator = _fit_calibrator(val_model_raw, y_val_model, method=calibration_method)
    explainer = shap.TreeExplainer(model)

    matrices = {
        "train": (X_train, y_train),
        "val_model": (X_val_model, y_val_model),
        "val_policy": (X_val_policy, y_val_policy),
        "test": (X_test, y_test),
    }
    metrics = _evaluate_model(model, calibrator, matrices)
    return TierArtifacts(
        tier=tier.upper(),
        builder=builder,
        model=model,
        calibrator=calibrator,
        explainer=explainer,
        metrics=metrics,
        feature_count=int(X_train.shape[1]),
        calibration_method=getattr(calibrator, "method", calibration_method),
    )


def _write_reproducibility_report(
    artifact_dir: str,
    report: dict[str, Any],
) -> str:
    resolved_dir = _resolve_project_path(artifact_dir)
    os.makedirs(resolved_dir, exist_ok=True)
    path = os.path.join(resolved_dir, REPRODUCIBILITY_REPORT_PATH)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return path


def train_and_serialize_models(
    *,
    processed_dir: str = DATA_DIR,
    artifact_dir: str = ARTIFACT_DIR,
    raw_dir: str = "data/raw/",
    allow_reduced_only: bool = False,
    calibration_method: str | None = None,
) -> dict[str, Any]:
    resolved_processed_dir = _resolve_project_path(processed_dir)
    resolved_artifact_dir = _resolve_project_path(artifact_dir)
    resolved_raw_dir = _resolve_project_path(raw_dir)
    splits = _load_processed_splits(resolved_processed_dir)
    chosen_calibration_method = (
        calibration_method
        or os.environ.get("MODEL_CALIBRATION_METHOD", DEFAULT_CALIBRATION_METHOD)
    ).strip().lower()

    reduced_artifacts = _build_tier_artifacts(
        tier="REDUCED",
        splits=splits,
        build_fn=build_reduced,
        fit_builder_fn=fit_reduced_builder,
        calibration_method=chosen_calibration_method,
    )
    _save_joblib(reduced_artifacts.model, resolved_artifact_dir, REDUCED_MODEL_PATH)
    _save_joblib(reduced_artifacts.calibrator, resolved_artifact_dir, REDUCED_CALIBRATOR_PATH)
    _save_joblib(reduced_artifacts.explainer, resolved_artifact_dir, REDUCED_EXPLAINER_PATH)

    full_artifacts: TierArtifacts | None = None
    if _has_full_raw_tables(resolved_raw_dir):
        full_artifacts = _build_tier_artifacts(
            tier="FULL",
            splits=splits,
            build_fn=build_full,
            fit_builder_fn=fit_full_builder,
            raw_dir=resolved_raw_dir,
            calibration_method=chosen_calibration_method,
        )
        _save_joblib(full_artifacts.model, resolved_artifact_dir, FULL_MODEL_PATH)
        _save_joblib(full_artifacts.calibrator, resolved_artifact_dir, FULL_CALIBRATOR_PATH)
        _save_joblib(full_artifacts.explainer, resolved_artifact_dir, FULL_EXPLAINER_PATH)
    elif not allow_reduced_only:
        raise FileNotFoundError(
            "FULL training could not run because data/raw/ is missing the required child tables. "
            "Set allow_reduced_only=True only if you intentionally want a reduced-only offline run."
        )

    deployed_version = (
        MODEL_VERSIONS["full"] if full_artifacts is not None else MODEL_VERSIONS["reduced"]
    )
    reproducibility_report = {
        "deployed_model_version": deployed_version,
        "champion_model_version": deployed_version,
        "random_state": RANDOM_STATE,
        "processed_dir": resolved_processed_dir,
        "artifact_dir": resolved_artifact_dir,
        "raw_dir": resolved_raw_dir,
        "allow_reduced_only": bool(allow_reduced_only),
        "training_profile": {
            "xgboost": dict(XGB_TRAINING_PROFILE),
            "calibration_method": chosen_calibration_method,
            "classification_threshold": DEFAULT_CLASSIFICATION_THRESHOLD,
        },
        "tiers": {
            "reduced": {
                "model_version": MODEL_VERSIONS["reduced"],
                "calibration_method": reduced_artifacts.calibration_method,
                "feature_count": reduced_artifacts.feature_count,
                "metrics": reduced_artifacts.metrics,
            },
            "full": None
            if full_artifacts is None
            else {
                "model_version": MODEL_VERSIONS["full"],
                "calibration_method": full_artifacts.calibration_method,
                "feature_count": full_artifacts.feature_count,
                "metrics": full_artifacts.metrics,
            },
        },
    }
    _write_reproducibility_report(resolved_artifact_dir, reproducibility_report)
    return reproducibility_report


def load_artifacts(artifact_dir: str = ARTIFACT_DIR) -> dict[str, Any]:
    resolved_artifact_dir = _resolve_project_path(artifact_dir)
    return {
        "full_model": joblib.load(os.path.join(resolved_artifact_dir, FULL_MODEL_PATH)),
        "reduced_model": joblib.load(os.path.join(resolved_artifact_dir, REDUCED_MODEL_PATH)),
        "full_calibrator": joblib.load(os.path.join(resolved_artifact_dir, FULL_CALIBRATOR_PATH)),
        "reduced_calibrator": joblib.load(os.path.join(resolved_artifact_dir, REDUCED_CALIBRATOR_PATH)),
        "full_shap_explainer": joblib.load(os.path.join(resolved_artifact_dir, FULL_EXPLAINER_PATH)),
        "reduced_shap_explainer": joblib.load(os.path.join(resolved_artifact_dir, REDUCED_EXPLAINER_PATH)),
    }


if __name__ == "__main__":
    allow_reduced_only = os.environ.get("ALLOW_REDUCED_ONLY", "0") == "1"
    report = train_and_serialize_models(
        processed_dir=os.environ.get("DATA_PROCESSED_DIR", DATA_DIR),
        artifact_dir=os.environ.get("ARTIFACT_DIR", ARTIFACT_DIR),
        raw_dir=os.environ.get("DATA_RAW_DIR", "data/raw/"),
        allow_reduced_only=allow_reduced_only,
        calibration_method=os.environ.get("MODEL_CALIBRATION_METHOD", DEFAULT_CALIBRATION_METHOD),
    )
    print(json.dumps(report, indent=2))
