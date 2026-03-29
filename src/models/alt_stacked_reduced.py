from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from configs.config import ARTIFACT_DIR, DATA_DIR
from src.feature_engineering import build_reduced, fit_reduced_builder
from src.models.alt_stacked_reduced_sidecars import (
    SK_ID_CURR_COL,
    align_sidecar_to_ids,
    extract_sidecar_feature_columns,
    load_ecommerce_sidecar,
    load_telco_sidecar,
    load_wallet_sidecar,
)
from src.models.train import (
    DatasetBundle,
    _candidate_model_params,
    _has_real_training_inputs,
    _load_real_bundle,
    _make_model,
    _select_calibrator,
    load_artifacts,
)

ALT_STACKED_REDUCED_DIRNAME = "alt_stacked_reduced"
TELCO_SUBMODEL_FILENAME = "telco_submodel.joblib"
TELCO_SUBMODEL_REPORT_FILENAME = "telco_submodel_report.json"
TELCO_OOF_PREDICTIONS_FILENAME = "telco_submodel_oof_train_predictions.csv"
TELCO_VAL_MODEL_PREDICTIONS_FILENAME = "telco_submodel_val_model_predictions.csv"
TELCO_VAL_POLICY_PREDICTIONS_FILENAME = "telco_submodel_val_policy_predictions.csv"
TELCO_TEST_PREDICTIONS_FILENAME = "telco_submodel_test_predictions.csv"
TELCO_SCORE_COL = "TELCO_SUBMODEL_SCORE"
WALLET_SUBMODEL_FILENAME = "wallet_submodel.joblib"
WALLET_SUBMODEL_REPORT_FILENAME = "wallet_submodel_report.json"
WALLET_OOF_PREDICTIONS_FILENAME = "wallet_submodel_oof_train_predictions.csv"
WALLET_VAL_MODEL_PREDICTIONS_FILENAME = "wallet_submodel_val_model_predictions.csv"
WALLET_VAL_POLICY_PREDICTIONS_FILENAME = "wallet_submodel_val_policy_predictions.csv"
WALLET_TEST_PREDICTIONS_FILENAME = "wallet_submodel_test_predictions.csv"
WALLET_SCORE_COL = "WALLET_SUBMODEL_SCORE"
ECOMMERCE_SUBMODEL_FILENAME = "ecommerce_submodel.joblib"
ECOMMERCE_SUBMODEL_REPORT_FILENAME = "ecommerce_submodel_report.json"
ECOMMERCE_OOF_PREDICTIONS_FILENAME = "ecommerce_submodel_oof_train_predictions.csv"
ECOMMERCE_VAL_MODEL_PREDICTIONS_FILENAME = "ecommerce_submodel_val_model_predictions.csv"
ECOMMERCE_VAL_POLICY_PREDICTIONS_FILENAME = "ecommerce_submodel_val_policy_predictions.csv"
ECOMMERCE_TEST_PREDICTIONS_FILENAME = "ecommerce_submodel_test_predictions.csv"
ECOMMERCE_SCORE_COL = "ECOMMERCE_SUBMODEL_SCORE"
META_SCORE_TELCO_COL = "META_SCORE_TELCO"
META_SCORE_WALLET_COL = "META_SCORE_WALLET"
META_SCORE_ECOMMERCE_COL = "META_SCORE_ECOMMERCE"
META_SCORE_LINEAGE_FILENAME = "meta_score_lineage.json"
STACKED_REDUCED_SPLIT_SUMMARY_FILENAME = "stacked_reduced_split_summary.json"
TRAIN_META_SCORES_FILENAME = "train_meta_scores.csv"
VAL_MODEL_META_SCORES_FILENAME = "val_model_meta_scores.csv"
VAL_POLICY_META_SCORES_FILENAME = "val_policy_meta_scores.csv"
TEST_META_SCORES_FILENAME = "test_meta_scores.csv"
STACKED_PROCESSED_DIRNAME = "alt_stacked_reduced"
STACKED_TRAIN_PKL_FILENAME = "train_stacked_reduced.pkl"
STACKED_VAL_MODEL_PKL_FILENAME = "val_model_stacked_reduced.pkl"
STACKED_VAL_POLICY_PKL_FILENAME = "val_policy_stacked_reduced.pkl"
STACKED_TEST_PKL_FILENAME = "test_stacked_reduced.pkl"
PROCESSED_MANIFEST_FILENAME = "processed_artifact_manifest.json"
MASTER_XGB_MODEL_FILENAME = "alt_stacked_reduced_master_xgb.joblib"
MASTER_XGB_CALIBRATOR_FILENAME = "alt_stacked_reduced_master_xgb_calibrator.joblib"
MASTER_XGB_REPORT_FILENAME = "alt_stacked_reduced_master_xgb_report.json"
MASTER_XGB_VAL_MODEL_PREDICTIONS_FILENAME = "alt_stacked_reduced_master_xgb_predictions_val_model.csv"
MASTER_XGB_VAL_POLICY_PREDICTIONS_FILENAME = "alt_stacked_reduced_master_xgb_predictions_val_policy.csv"
MASTER_XGB_TEST_PREDICTIONS_FILENAME = "alt_stacked_reduced_master_xgb_predictions_test.csv"
MASTER_XGB_RAW_SCORE_COL = "ALT_STACKED_REDUCED_MASTER_XGB_SCORE_RAW"
MASTER_XGB_CALIBRATED_SCORE_COL = "ALT_STACKED_REDUCED_MASTER_XGB_SCORE_CALIBRATED"
ABLATION_REPORT_FILENAME = "alt_stacked_reduced_ablation_report.json"
ABLATION_SUMMARY_FILENAME = "alt_stacked_reduced_ablation_summary.md"
DEFAULT_SUBMODEL_OOF_FOLDS = 5
ALL_META_SCORE_COLS = [
    META_SCORE_TELCO_COL,
    META_SCORE_WALLET_COL,
    META_SCORE_ECOMMERCE_COL,
]
ABLATION_CONFIG_META_COLUMNS = {
    "BASE_REDUCED_ONLY": [],
    "REDUCED_PLUS_TELCO": [META_SCORE_TELCO_COL],
    "REDUCED_PLUS_WALLET": [META_SCORE_WALLET_COL],
    "REDUCED_PLUS_ECOMMERCE": [META_SCORE_ECOMMERCE_COL],
    "REDUCED_PLUS_TELCO_WALLET": [META_SCORE_TELCO_COL, META_SCORE_WALLET_COL],
    "REDUCED_PLUS_ALL_THREE": [
        META_SCORE_TELCO_COL,
        META_SCORE_WALLET_COL,
        META_SCORE_ECOMMERCE_COL,
    ],
}

__all__ = [
    "ECOMMERCE_SCORE_COL",
    "META_SCORE_ECOMMERCE_COL",
    "META_SCORE_TELCO_COL",
    "META_SCORE_WALLET_COL",
    "TELCO_SCORE_COL",
    "WALLET_SCORE_COL",
    "align_ecommerce_sidecar_to_bundle",
    "align_telco_sidecar_to_bundle",
    "align_wallet_sidecar_to_bundle",
    "append_meta_scores_to_reduced_matrix",
    "build_ablation_stacked_matrices_for_config",
    "build_ablation_stacked_matrix_for_config",
    "build_all_meta_score_frames",
    "build_canonical_reduced_matrices_for_bundle",
    "build_meta_score_frame_for_split",
    "build_stacked_reduced_split_matrices",
    "calibrate_alt_stacked_reduced_master",
    "evaluate_alt_stacked_reduced_master",
    "extract_X_y_from_stacked_split",
    "fit_alt_stacked_reduced_master_xgb",
    "extract_ecommerce_feature_columns",
    "extract_telco_feature_columns",
    "extract_wallet_feature_columns",
    "fit_ecommerce_submodel",
    "fit_telco_submodel",
    "fit_wallet_submodel",
    "generate_ecommerce_oof_predictions",
    "generate_ecommerce_prediction_frame",
    "generate_telco_oof_predictions",
    "generate_telco_prediction_frame",
    "generate_wallet_oof_predictions",
    "generate_wallet_prediction_frame",
    "load_alt_stacked_reduced_bundle",
    "load_frozen_reduced_baseline_metrics",
    "load_ecommerce_prediction_tables",
    "load_stacked_reduced_split_matrices",
    "load_telco_prediction_tables",
    "load_wallet_prediction_tables",
    "load_and_align_ecommerce_sidecar_to_bundle",
    "load_and_align_telco_sidecar_to_bundle",
    "load_and_align_wallet_sidecar_to_bundle",
    "run_alt_stacked_reduced_master_xgb_experiment",
    "run_alt_stacked_reduced_ablation_experiment",
    "run_meta_score_assembly_experiment",
    "run_ecommerce_submodel_experiment",
    "run_telco_submodel_experiment",
    "run_wallet_submodel_experiment",
    "validate_side_prediction_frame",
]


class ConstantProbabilityModel:
    def __init__(self, probability: float):
        self.probability = float(np.clip(probability, 0.0, 1.0))

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if hasattr(X, "__len__"):
            n_rows = len(X)  # type: ignore[arg-type]
        else:
            n_rows = int(np.asarray(X).shape[0])
        probs = np.full(int(n_rows), self.probability, dtype=float)
        return np.column_stack([1.0 - probs, probs])


def _alt_artifact_dir(artifact_dir: str) -> str:
    return os.path.join(artifact_dir, ALT_STACKED_REDUCED_DIRNAME)


def _artifact_path(artifact_dir: str, filename: str) -> str:
    return os.path.join(artifact_dir, filename)


def _stacked_processed_dir(processed_dir: str) -> str:
    return os.path.join(processed_dir, STACKED_PROCESSED_DIRNAME)


def _write_json(path: str, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _prediction_path_map(
    artifact_dir: str,
    *,
    train_oof_filename: str,
    val_model_filename: str,
    val_policy_filename: str,
    test_filename: str,
) -> dict[str, str]:
    return {
        "train_oof": _artifact_path(artifact_dir, train_oof_filename),
        "val_model": _artifact_path(artifact_dir, val_model_filename),
        "val_policy": _artifact_path(artifact_dir, val_policy_filename),
        "test": _artifact_path(artifact_dir, test_filename),
    }


def _meta_score_artifact_path_map(artifact_dir: str) -> dict[str, str]:
    return {
        "train": _artifact_path(artifact_dir, TRAIN_META_SCORES_FILENAME),
        "val_model": _artifact_path(artifact_dir, VAL_MODEL_META_SCORES_FILENAME),
        "val_policy": _artifact_path(artifact_dir, VAL_POLICY_META_SCORES_FILENAME),
        "test": _artifact_path(artifact_dir, TEST_META_SCORES_FILENAME),
    }


def _stacked_matrix_path_map(processed_dir: str) -> dict[str, str]:
    stacked_dir = _stacked_processed_dir(processed_dir)
    return {
        "train": os.path.join(stacked_dir, STACKED_TRAIN_PKL_FILENAME),
        "val_model": os.path.join(stacked_dir, STACKED_VAL_MODEL_PKL_FILENAME),
        "val_policy": os.path.join(stacked_dir, STACKED_VAL_POLICY_PKL_FILENAME),
        "test": os.path.join(stacked_dir, STACKED_TEST_PKL_FILENAME),
    }


def _master_prediction_path_map(artifact_dir: str) -> dict[str, str]:
    return {
        "val_model": _artifact_path(artifact_dir, MASTER_XGB_VAL_MODEL_PREDICTIONS_FILENAME),
        "val_policy": _artifact_path(artifact_dir, MASTER_XGB_VAL_POLICY_PREDICTIONS_FILENAME),
        "test": _artifact_path(artifact_dir, MASTER_XGB_TEST_PREDICTIONS_FILENAME),
    }


def _side_prediction_specs(artifact_dir: str) -> dict[str, dict[str, Any]]:
    experiment_artifact_dir = _alt_artifact_dir(artifact_dir)
    return {
        "telco": {
            "score_column": TELCO_SCORE_COL,
            "meta_column": META_SCORE_TELCO_COL,
            "paths": _prediction_path_map(
                experiment_artifact_dir,
                train_oof_filename=TELCO_OOF_PREDICTIONS_FILENAME,
                val_model_filename=TELCO_VAL_MODEL_PREDICTIONS_FILENAME,
                val_policy_filename=TELCO_VAL_POLICY_PREDICTIONS_FILENAME,
                test_filename=TELCO_TEST_PREDICTIONS_FILENAME,
            ),
        },
        "wallet": {
            "score_column": WALLET_SCORE_COL,
            "meta_column": META_SCORE_WALLET_COL,
            "paths": _prediction_path_map(
                experiment_artifact_dir,
                train_oof_filename=WALLET_OOF_PREDICTIONS_FILENAME,
                val_model_filename=WALLET_VAL_MODEL_PREDICTIONS_FILENAME,
                val_policy_filename=WALLET_VAL_POLICY_PREDICTIONS_FILENAME,
                test_filename=WALLET_TEST_PREDICTIONS_FILENAME,
            ),
        },
        "ecommerce": {
            "score_column": ECOMMERCE_SCORE_COL,
            "meta_column": META_SCORE_ECOMMERCE_COL,
            "paths": _prediction_path_map(
                experiment_artifact_dir,
                train_oof_filename=ECOMMERCE_OOF_PREDICTIONS_FILENAME,
                val_model_filename=ECOMMERCE_VAL_MODEL_PREDICTIONS_FILENAME,
                val_policy_filename=ECOMMERCE_VAL_POLICY_PREDICTIONS_FILENAME,
                test_filename=ECOMMERCE_TEST_PREDICTIONS_FILENAME,
            ),
        },
    }


def load_alt_stacked_reduced_bundle(
    processed_dir: str = DATA_DIR,
    raw_dir: str = "data/raw/",
) -> DatasetBundle:
    if not _has_real_training_inputs(processed_dir, raw_dir):
        raise RuntimeError("ALT_STACKED_REDUCED requires real processed splits and raw aggregate tables.")
    return _load_real_bundle(processed_dir, raw_dir)


def _align_sidecar_family_to_bundle(
    bundle: DatasetBundle,
    sidecar_df: pd.DataFrame,
    family_name: str,
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, Any]]]:
    aligned_frames: dict[str, pd.DataFrame] = {}
    alignment_summary: dict[str, dict[str, Any]] = {}
    for split_name in ("train", "val_model", "val_policy", "test"):
        split_df = getattr(bundle, split_name)
        aligned_frame, metadata = align_sidecar_to_ids(
            split_df[[SK_ID_CURR_COL]],
            sidecar_df,
            family_name,
        )
        aligned_frames[split_name] = aligned_frame
        alignment_summary[split_name] = metadata
    return aligned_frames, alignment_summary


def align_telco_sidecar_to_bundle(
    bundle: DatasetBundle,
    telco_sidecar_df: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, Any]]]:
    return _align_sidecar_family_to_bundle(bundle, telco_sidecar_df, "telco_utility")


def align_wallet_sidecar_to_bundle(
    bundle: DatasetBundle,
    wallet_sidecar_df: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, Any]]]:
    return _align_sidecar_family_to_bundle(bundle, wallet_sidecar_df, "wallet_p2p")


def align_ecommerce_sidecar_to_bundle(
    bundle: DatasetBundle,
    ecommerce_sidecar_df: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, Any]]]:
    return _align_sidecar_family_to_bundle(bundle, ecommerce_sidecar_df, "ecommerce_social")


def load_and_align_telco_sidecar_to_bundle(
    bundle: DatasetBundle,
    telco_sidecar_path: str | Path,
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, Any]], pd.DataFrame]:
    telco_sidecar_df = load_telco_sidecar(telco_sidecar_path)
    aligned_frames, alignment_summary = align_telco_sidecar_to_bundle(bundle, telco_sidecar_df)
    return aligned_frames, alignment_summary, telco_sidecar_df


def load_and_align_wallet_sidecar_to_bundle(
    bundle: DatasetBundle,
    wallet_sidecar_path: str | Path,
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, Any]], pd.DataFrame]:
    wallet_sidecar_df = load_wallet_sidecar(wallet_sidecar_path)
    aligned_frames, alignment_summary = align_wallet_sidecar_to_bundle(bundle, wallet_sidecar_df)
    return aligned_frames, alignment_summary, wallet_sidecar_df


def load_and_align_ecommerce_sidecar_to_bundle(
    bundle: DatasetBundle,
    ecommerce_sidecar_path: str | Path,
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, Any]], pd.DataFrame]:
    ecommerce_sidecar_df = load_ecommerce_sidecar(ecommerce_sidecar_path)
    aligned_frames, alignment_summary = align_ecommerce_sidecar_to_bundle(bundle, ecommerce_sidecar_df)
    return aligned_frames, alignment_summary, ecommerce_sidecar_df


def extract_telco_feature_columns(telco_sidecar_df: pd.DataFrame) -> list[str]:
    return extract_sidecar_feature_columns(telco_sidecar_df)


def extract_wallet_feature_columns(wallet_sidecar_df: pd.DataFrame) -> list[str]:
    return extract_sidecar_feature_columns(wallet_sidecar_df)


def extract_ecommerce_feature_columns(ecommerce_sidecar_df: pd.DataFrame) -> list[str]:
    return extract_sidecar_feature_columns(ecommerce_sidecar_df)


def _positive_rate(y: np.ndarray) -> float:
    if y.size == 0:
        return 0.5
    return float(np.mean(y))


def _fit_submodel_from_arrays(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    params: dict[str, Any],
) -> Any:
    unique_classes = np.unique(y_train)
    if len(y_train) == 0:
        return ConstantProbabilityModel(0.5)
    if unique_classes.size < 2:
        return ConstantProbabilityModel(_positive_rate(y_train))
    model = _make_model(params)
    model.fit(X_train, y_train, verbose=False)
    return model


def _predict_positive_class(model: Any, X: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict_proba(X), dtype=float)[:, 1]


def _safe_metric(metric_fn: Any, y_true: np.ndarray, scores: np.ndarray) -> float | None:
    try:
        return float(metric_fn(y_true, scores))
    except ValueError:
        return None


def _collect_submodel_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float | None]:
    return {
        "roc_auc": _safe_metric(roc_auc_score, y_true, scores),
        "pr_auc": _safe_metric(average_precision_score, y_true, scores),
        "brier_score": _safe_metric(brier_score_loss, y_true, scores),
        "default_rate": float(np.mean(y_true)) if len(y_true) else None,
    }


def _fit_side_submodel(
    train_X: pd.DataFrame,
    train_y: np.ndarray,
    val_model_X: pd.DataFrame,
    val_model_y: np.ndarray,
) -> dict[str, Any]:
    pos = int(train_y.sum())
    neg = int(len(train_y) - pos)
    scale_pos_weight = float(neg / max(pos, 1))

    best_model = None
    best_name = ""
    best_params: dict[str, Any] = {}
    best_scores: np.ndarray | None = None
    best_roc_auc = float("-inf")

    for candidate_name, params in _candidate_model_params(scale_pos_weight):
        model = _fit_submodel_from_arrays(train_X, train_y, params)
        val_scores = _predict_positive_class(model, val_model_X)
        val_metrics = _collect_submodel_metrics(val_model_y, val_scores)
        ranking_metric = val_metrics["roc_auc"] if val_metrics["roc_auc"] is not None else float("-inf")
        if ranking_metric > best_roc_auc:
            best_model = model
            best_name = candidate_name
            best_params = dict(params)
            best_scores = val_scores
            best_roc_auc = float(ranking_metric)

    assert best_model is not None and best_scores is not None
    return {
        "model": best_model,
        "selected_candidate": best_name,
        "selected_params": best_params,
        "val_model_scores": best_scores,
        "val_model_metrics": _collect_submodel_metrics(val_model_y, best_scores),
    }


def fit_telco_submodel(
    train_X: pd.DataFrame,
    train_y: np.ndarray,
    val_model_X: pd.DataFrame,
    val_model_y: np.ndarray,
) -> dict[str, Any]:
    return _fit_side_submodel(train_X, train_y, val_model_X, val_model_y)


def fit_wallet_submodel(
    train_X: pd.DataFrame,
    train_y: np.ndarray,
    val_model_X: pd.DataFrame,
    val_model_y: np.ndarray,
) -> dict[str, Any]:
    return _fit_side_submodel(train_X, train_y, val_model_X, val_model_y)


def fit_ecommerce_submodel(
    train_X: pd.DataFrame,
    train_y: np.ndarray,
    val_model_X: pd.DataFrame,
    val_model_y: np.ndarray,
) -> dict[str, Any]:
    return _fit_side_submodel(train_X, train_y, val_model_X, val_model_y)


def _iter_blocked_oof_folds(n_rows: int, n_folds: int = DEFAULT_SUBMODEL_OOF_FOLDS) -> list[dict[str, int]]:
    if n_rows <= 0:
        return []
    effective_folds = max(1, min(int(n_folds), n_rows))
    index_blocks = [block for block in np.array_split(np.arange(n_rows, dtype=int), effective_folds) if block.size]
    folds: list[dict[str, int]] = []
    for fold_index, block in enumerate(index_blocks):
        folds.append(
            {
                "fold_index": int(fold_index),
                "val_start": int(block[0]),
                "val_end_exclusive": int(block[-1] + 1),
            }
        )
    return folds


def _generate_side_oof_predictions(
    train_ids: pd.Series,
    train_X: pd.DataFrame,
    train_y: np.ndarray,
    selected_params: dict[str, Any],
    *,
    score_column: str,
    n_folds: int = DEFAULT_SUBMODEL_OOF_FOLDS,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if len(train_ids) != len(train_X) or len(train_X) != len(train_y):
        raise ValueError("train_ids, train_X, and train_y must have identical row counts")

    oof_scores = np.full(len(train_y), np.nan, dtype=float)
    fold_summaries: list[dict[str, Any]] = []

    for fold in _iter_blocked_oof_folds(len(train_y), n_folds=n_folds):
        val_start = fold["val_start"]
        val_end = fold["val_end_exclusive"]
        X_val = train_X.iloc[val_start:val_end].reset_index(drop=True)
        y_val = train_y[val_start:val_end]
        X_fit = train_X.iloc[:val_start].reset_index(drop=True)
        y_fit = train_y[:val_start]

        if len(y_fit) == 0:
            model = ConstantProbabilityModel(0.5)
            fit_mode = "constant_prior_no_history"
        elif np.unique(y_fit).size < 2:
            model = ConstantProbabilityModel(_positive_rate(y_fit))
            fit_mode = "constant_prior_single_class_history"
        else:
            model = _fit_submodel_from_arrays(X_fit, y_fit, selected_params)
            fit_mode = "xgboost_prefix_fit"

        fold_scores = _predict_positive_class(model, X_val)
        oof_scores[val_start:val_end] = fold_scores
        fold_summaries.append(
            {
                "fold_index": fold["fold_index"],
                "fit_row_count": int(len(X_fit)),
                "validation_row_count": int(len(X_val)),
                "validation_start_row": int(val_start),
                "validation_end_row_exclusive": int(val_end),
                "fit_mode": fit_mode,
                "validation_metrics": _collect_submodel_metrics(y_val, fold_scores),
            }
        )

    if np.isnan(oof_scores).any():
        raise RuntimeError("Submodel OOF generation did not assign scores to every train row")

    prediction_df = pd.DataFrame(
        {
            SK_ID_CURR_COL: train_ids.to_numpy(copy=False),
            score_column: oof_scores,
            "split_name": "train_oof",
        }
    )
    metadata = {
        "n_folds": int(n_folds),
        "folds": fold_summaries,
        "metrics": _collect_submodel_metrics(train_y, oof_scores),
    }
    return prediction_df, metadata


def generate_telco_oof_predictions(
    train_ids: pd.Series,
    train_X: pd.DataFrame,
    train_y: np.ndarray,
    selected_params: dict[str, Any],
    n_folds: int = DEFAULT_SUBMODEL_OOF_FOLDS,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    return _generate_side_oof_predictions(
        train_ids,
        train_X,
        train_y,
        selected_params,
        score_column=TELCO_SCORE_COL,
        n_folds=n_folds,
    )


def generate_wallet_oof_predictions(
    train_ids: pd.Series,
    train_X: pd.DataFrame,
    train_y: np.ndarray,
    selected_params: dict[str, Any],
    n_folds: int = DEFAULT_SUBMODEL_OOF_FOLDS,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    return _generate_side_oof_predictions(
        train_ids,
        train_X,
        train_y,
        selected_params,
        score_column=WALLET_SCORE_COL,
        n_folds=n_folds,
    )


def generate_ecommerce_oof_predictions(
    train_ids: pd.Series,
    train_X: pd.DataFrame,
    train_y: np.ndarray,
    selected_params: dict[str, Any],
    n_folds: int = DEFAULT_SUBMODEL_OOF_FOLDS,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    return _generate_side_oof_predictions(
        train_ids,
        train_X,
        train_y,
        selected_params,
        score_column=ECOMMERCE_SCORE_COL,
        n_folds=n_folds,
    )


def _generate_side_prediction_frame(
    split_ids: pd.Series,
    split_X: pd.DataFrame,
    split_name: str,
    model: Any,
    *,
    score_column: str,
) -> pd.DataFrame:
    scores = _predict_positive_class(model, split_X)
    return pd.DataFrame(
        {
            SK_ID_CURR_COL: split_ids.to_numpy(copy=False),
            score_column: scores,
            "split_name": split_name,
        }
    )


def generate_telco_prediction_frame(
    split_ids: pd.Series,
    split_X: pd.DataFrame,
    split_name: str,
    model: Any,
) -> pd.DataFrame:
    return _generate_side_prediction_frame(
        split_ids,
        split_X,
        split_name,
        model,
        score_column=TELCO_SCORE_COL,
    )


def generate_wallet_prediction_frame(
    split_ids: pd.Series,
    split_X: pd.DataFrame,
    split_name: str,
    model: Any,
) -> pd.DataFrame:
    return _generate_side_prediction_frame(
        split_ids,
        split_X,
        split_name,
        model,
        score_column=WALLET_SCORE_COL,
    )


def generate_ecommerce_prediction_frame(
    split_ids: pd.Series,
    split_X: pd.DataFrame,
    split_name: str,
    model: Any,
) -> pd.DataFrame:
    return _generate_side_prediction_frame(
        split_ids,
        split_X,
        split_name,
        model,
        score_column=ECOMMERCE_SCORE_COL,
    )


def _write_prediction_frame(path: str, prediction_df: pd.DataFrame, *, score_column: str) -> None:
    required_columns = [SK_ID_CURR_COL, score_column, "split_name"]
    if list(prediction_df.columns) != required_columns:
        raise ValueError(f"Prediction frame must contain columns {required_columns}")
    if prediction_df[SK_ID_CURR_COL].duplicated().any():
        raise ValueError("Prediction frame contains duplicate SK_ID_CURR values")
    prediction_df.to_csv(path, index=False)


def validate_side_prediction_frame(
    prediction_df: pd.DataFrame,
    *,
    split_name: str,
    expected_split_name: str,
    score_column: str,
    expected_ids: pd.Series | pd.Index | np.ndarray | list[int],
) -> pd.DataFrame:
    required_columns = [SK_ID_CURR_COL, score_column, "split_name"]
    if list(prediction_df.columns) != required_columns:
        raise ValueError(f"{split_name} prediction frame must contain columns {required_columns}")
    if prediction_df[SK_ID_CURR_COL].duplicated().any():
        raise ValueError(f"{split_name} prediction frame contains duplicate SK_ID_CURR values")
    if prediction_df[score_column].isna().any():
        raise ValueError(f"{split_name} prediction frame contains missing {score_column} values")
    if prediction_df["split_name"].nunique() != 1:
        raise ValueError(f"{split_name} prediction frame must contain exactly one split_name value")
    actual_split_name = str(prediction_df["split_name"].iloc[0])
    if actual_split_name != expected_split_name:
        raise ValueError(
            f"{split_name} prediction frame has split_name={actual_split_name!r}, expected {expected_split_name!r}"
        )

    expected_id_list = list(pd.Index(expected_ids).tolist())
    actual_id_list = prediction_df[SK_ID_CURR_COL].tolist()
    if actual_id_list != expected_id_list:
        raise ValueError(f"{split_name} prediction frame SK_ID_CURR order does not match the canonical split order")
    return prediction_df


def _load_side_prediction_tables(
    *,
    artifact_dir: str,
    bundle: DatasetBundle,
    side_label: str,
) -> dict[str, pd.DataFrame]:
    side_spec = _side_prediction_specs(artifact_dir)[side_label]
    score_column = str(side_spec["score_column"])
    paths: dict[str, str] = dict(side_spec["paths"])
    expected_split_names = {
        "train": "train_oof",
        "val_model": "val_model",
        "val_policy": "val_policy",
        "test": "test",
    }

    tables: dict[str, pd.DataFrame] = {}
    for split_name, source_key in [
        ("train", "train_oof"),
        ("val_model", "val_model"),
        ("val_policy", "val_policy"),
        ("test", "test"),
    ]:
        path = paths[source_key]
        if not os.path.exists(path):
            raise RuntimeError(f"Missing required {side_label} prediction artifact for {split_name}: {path}")
        frame = pd.read_csv(path)
        tables[split_name] = validate_side_prediction_frame(
            frame,
            split_name=split_name,
            expected_split_name=expected_split_names[split_name],
            score_column=score_column,
            expected_ids=getattr(bundle, split_name)[SK_ID_CURR_COL],
        )
    return tables


def load_telco_prediction_tables(
    bundle: DatasetBundle,
    artifact_dir: str = ARTIFACT_DIR,
) -> dict[str, pd.DataFrame]:
    return _load_side_prediction_tables(artifact_dir=artifact_dir, bundle=bundle, side_label="telco")


def load_wallet_prediction_tables(
    bundle: DatasetBundle,
    artifact_dir: str = ARTIFACT_DIR,
) -> dict[str, pd.DataFrame]:
    return _load_side_prediction_tables(artifact_dir=artifact_dir, bundle=bundle, side_label="wallet")


def load_ecommerce_prediction_tables(
    bundle: DatasetBundle,
    artifact_dir: str = ARTIFACT_DIR,
) -> dict[str, pd.DataFrame]:
    return _load_side_prediction_tables(artifact_dir=artifact_dir, bundle=bundle, side_label="ecommerce")


def build_meta_score_frame_for_split(
    master_split_df: pd.DataFrame,
    split_name: str,
    telco_prediction_df: pd.DataFrame,
    wallet_prediction_df: pd.DataFrame,
    ecommerce_prediction_df: pd.DataFrame,
) -> pd.DataFrame:
    master = master_split_df[[SK_ID_CURR_COL]].copy().reset_index(drop=True)
    meta = master.merge(
        telco_prediction_df[[SK_ID_CURR_COL, TELCO_SCORE_COL]].rename(columns={TELCO_SCORE_COL: META_SCORE_TELCO_COL}),
        on=SK_ID_CURR_COL,
        how="left",
        sort=False,
        validate="one_to_one",
    )
    meta = meta.merge(
        wallet_prediction_df[[SK_ID_CURR_COL, WALLET_SCORE_COL]].rename(columns={WALLET_SCORE_COL: META_SCORE_WALLET_COL}),
        on=SK_ID_CURR_COL,
        how="left",
        sort=False,
        validate="one_to_one",
    )
    meta = meta.merge(
        ecommerce_prediction_df[[SK_ID_CURR_COL, ECOMMERCE_SCORE_COL]].rename(columns={ECOMMERCE_SCORE_COL: META_SCORE_ECOMMERCE_COL}),
        on=SK_ID_CURR_COL,
        how="left",
        sort=False,
        validate="one_to_one",
    )
    meta["split_name"] = split_name

    required_columns = [
        SK_ID_CURR_COL,
        META_SCORE_TELCO_COL,
        META_SCORE_WALLET_COL,
        META_SCORE_ECOMMERCE_COL,
        "split_name",
    ]
    if list(meta.columns) != required_columns:
        raise ValueError(f"{split_name} meta-score frame must contain columns {required_columns}")
    if meta[SK_ID_CURR_COL].duplicated().any():
        raise ValueError(f"{split_name} meta-score frame contains duplicate SK_ID_CURR values")
    missing_meta_cols = [
        column
        for column in [META_SCORE_TELCO_COL, META_SCORE_WALLET_COL, META_SCORE_ECOMMERCE_COL]
        if meta[column].isna().any()
    ]
    if missing_meta_cols:
        raise ValueError(f"{split_name} meta-score frame contains missing values for columns: {missing_meta_cols}")
    return meta


def build_all_meta_score_frames(
    bundle: DatasetBundle,
    telco_tables: dict[str, pd.DataFrame],
    wallet_tables: dict[str, pd.DataFrame],
    ecommerce_tables: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    return {
        "train": build_meta_score_frame_for_split(
            bundle.train,
            "train",
            telco_tables["train"],
            wallet_tables["train"],
            ecommerce_tables["train"],
        ),
        "val_model": build_meta_score_frame_for_split(
            bundle.val_model,
            "val_model",
            telco_tables["val_model"],
            wallet_tables["val_model"],
            ecommerce_tables["val_model"],
        ),
        "val_policy": build_meta_score_frame_for_split(
            bundle.val_policy,
            "val_policy",
            telco_tables["val_policy"],
            wallet_tables["val_policy"],
            ecommerce_tables["val_policy"],
        ),
        "test": build_meta_score_frame_for_split(
            bundle.test,
            "test",
            telco_tables["test"],
            wallet_tables["test"],
            ecommerce_tables["test"],
        ),
    }


def build_canonical_reduced_matrices_for_bundle(
    bundle: DatasetBundle,
    *,
    processed_dir: str = DATA_DIR,
) -> dict[str, Any]:
    processed_manifest_path = os.path.join(processed_dir, PROCESSED_MANIFEST_FILENAME)
    resolved_manifest_path = processed_manifest_path if os.path.exists(processed_manifest_path) else None
    reduced_builder = fit_reduced_builder(
        bundle.train,
        save_path=None,
        processed_manifest_path=resolved_manifest_path,
    )
    matrices = {
        "train": build_reduced(bundle.train, reduced_builder),
        "val_model": build_reduced(bundle.val_model, reduced_builder),
        "val_policy": build_reduced(bundle.val_policy, reduced_builder),
        "test": build_reduced(bundle.test, reduced_builder),
    }
    return {
        "builder": reduced_builder,
        "matrices": matrices,
    }


def append_meta_scores_to_reduced_matrix(
    reduced_matrix: pd.DataFrame,
    meta_scores_df: pd.DataFrame,
) -> pd.DataFrame:
    missing_meta = [column for column in ALL_META_SCORE_COLS if column not in meta_scores_df.columns]
    if missing_meta:
        raise ValueError(f"Meta-score frame is missing required columns: {missing_meta}")
    if len(reduced_matrix) != len(meta_scores_df):
        raise ValueError("Reduced matrix and meta-score frame must have identical row counts")
    meta_only = meta_scores_df[ALL_META_SCORE_COLS].reset_index(drop=True)
    return pd.concat([reduced_matrix.reset_index(drop=True), meta_only], axis=1)


def build_stacked_reduced_split_matrices(
    bundle: DatasetBundle,
    meta_score_frames: dict[str, pd.DataFrame],
    *,
    processed_dir: str = DATA_DIR,
) -> dict[str, Any]:
    canonical = build_canonical_reduced_matrices_for_bundle(bundle, processed_dir=processed_dir)
    canonical_matrices: dict[str, pd.DataFrame] = canonical["matrices"]
    stacked_matrices = {
        split_name: append_meta_scores_to_reduced_matrix(canonical_matrices[split_name], meta_score_frames[split_name])
        for split_name in ("train", "val_model", "val_policy", "test")
    }
    return {
        "builder": canonical["builder"],
        "canonical_matrices": canonical_matrices,
        "stacked_matrices": stacked_matrices,
    }


def load_stacked_reduced_split_matrices(
    processed_dir: str = DATA_DIR,
) -> dict[str, pd.DataFrame]:
    paths = _stacked_matrix_path_map(processed_dir)
    matrices: dict[str, pd.DataFrame] = {}
    for split_name, path in paths.items():
        if not os.path.exists(path):
            raise RuntimeError(f"Missing required stacked REDUCED matrix for {split_name}: {path}")
        matrices[split_name] = pd.read_pickle(path)
    return matrices


def build_ablation_stacked_matrix_for_config(
    stacked_split_df: pd.DataFrame,
    selected_meta_columns: list[str],
) -> pd.DataFrame:
    missing_meta = [column for column in selected_meta_columns if column not in stacked_split_df.columns]
    if missing_meta:
        raise ValueError(f"Stacked matrix is missing required meta columns for ablation: {missing_meta}")
    canonical_columns = [column for column in stacked_split_df.columns if column not in ALL_META_SCORE_COLS]
    ordered_meta_columns = [column for column in ALL_META_SCORE_COLS if column in selected_meta_columns]
    selected_columns = canonical_columns + ordered_meta_columns
    return stacked_split_df[selected_columns].copy()


def build_ablation_stacked_matrices_for_config(
    stacked_matrices: dict[str, pd.DataFrame],
    selected_meta_columns: list[str],
) -> dict[str, pd.DataFrame]:
    return {
        split_name: build_ablation_stacked_matrix_for_config(stacked_matrices[split_name], selected_meta_columns)
        for split_name in ("train", "val_model", "val_policy", "test")
    }


def extract_X_y_from_stacked_split(
    bundle: DatasetBundle,
    stacked_split_df: pd.DataFrame,
    split_name: str,
) -> tuple[pd.Series, pd.DataFrame, np.ndarray]:
    split_df = getattr(bundle, split_name)
    if len(stacked_split_df) != len(split_df):
        raise ValueError(f"{split_name} stacked split row count does not match the canonical bundle split")
    ids = split_df[SK_ID_CURR_COL].reset_index(drop=True)
    y = split_df["TARGET"].to_numpy(dtype=int)
    X = stacked_split_df.reset_index(drop=True).copy()
    return ids, X, y


def fit_alt_stacked_reduced_master_xgb(
    train_X: pd.DataFrame,
    train_y: np.ndarray,
    val_model_X: pd.DataFrame,
    val_model_y: np.ndarray,
) -> dict[str, Any]:
    return _fit_side_submodel(train_X, train_y, val_model_X, val_model_y)


def calibrate_alt_stacked_reduced_master(
    val_policy_y: np.ndarray,
    val_policy_raw_scores: np.ndarray,
) -> dict[str, Any]:
    calibrator, calibrator_name, calibrator_metrics = _select_calibrator(val_policy_y, val_policy_raw_scores)
    calibrated_scores = np.asarray(calibrator.predict(val_policy_raw_scores), dtype=float)
    return {
        "calibrator": calibrator,
        "selected_calibrator": calibrator_name,
        "val_policy_calibration_metrics": calibrator_metrics,
        "val_policy_calibrated_scores": calibrated_scores,
    }


def evaluate_alt_stacked_reduced_master(
    y_true: np.ndarray,
    raw_scores: np.ndarray,
    calibrated_scores: np.ndarray | None = None,
) -> dict[str, Any]:
    payload = {
        "raw": _collect_submodel_metrics(y_true, raw_scores),
    }
    if calibrated_scores is not None:
        payload["calibrated"] = _collect_submodel_metrics(y_true, calibrated_scores)
    return payload


def _build_master_prediction_frame(
    split_ids: pd.Series,
    split_name: str,
    raw_scores: np.ndarray,
    calibrated_scores: np.ndarray | None,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            SK_ID_CURR_COL: split_ids.to_numpy(copy=False),
            MASTER_XGB_RAW_SCORE_COL: np.asarray(raw_scores, dtype=float),
            MASTER_XGB_CALIBRATED_SCORE_COL: (
                np.asarray(calibrated_scores, dtype=float)
                if calibrated_scores is not None
                else np.asarray(raw_scores, dtype=float)
            ),
            "split_name": split_name,
        }
    )
    if frame[SK_ID_CURR_COL].duplicated().any():
        raise ValueError(f"{split_name} master prediction frame contains duplicate SK_ID_CURR values")
    return frame


def _write_master_prediction_frame(path: str, frame: pd.DataFrame) -> None:
    required_columns = [
        SK_ID_CURR_COL,
        MASTER_XGB_RAW_SCORE_COL,
        MASTER_XGB_CALIBRATED_SCORE_COL,
        "split_name",
    ]
    if list(frame.columns) != required_columns:
        raise ValueError(f"Master prediction frame must contain columns {required_columns}")
    frame.to_csv(path, index=False)


def _comparison_vs_frozen_reduced(
    *,
    frozen_baseline: dict[str, Any],
    val_model_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "frozen_reduced_val_model_roc_auc": frozen_baseline["val_model"]["raw"]["roc_auc"],
        "frozen_reduced_test_roc_auc": frozen_baseline["test"]["calibrated"]["roc_auc"],
        "frozen_reduced_brier": frozen_baseline["test"]["calibrated"]["brier_score"],
        "delta_vs_frozen_reduced": {
            "val_model_roc_auc": (
                (val_model_metrics["raw"]["roc_auc"] or 0.0)
                - (frozen_baseline["val_model"]["raw"]["roc_auc"] or 0.0)
            ),
            "test_roc_auc": (
                (test_metrics["calibrated"]["roc_auc"] or 0.0)
                - (frozen_baseline["test"]["calibrated"]["roc_auc"] or 0.0)
            ),
            "test_pr_auc": (
                (test_metrics["calibrated"]["pr_auc"] or 0.0)
                - (frozen_baseline["test"]["calibrated"]["pr_auc"] or 0.0)
            ),
            "calibrated_test_brier": (
                (test_metrics["calibrated"]["brier_score"] or 0.0)
                - (frozen_baseline["test"]["calibrated"]["brier_score"] or 0.0)
            ),
        },
        "baseline_source": frozen_baseline["source"],
    }


def _summarize_ablation_recommendation(
    configurations: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    best_roc_config = max(
        configurations.items(),
        key=lambda item: item[1]["metrics"]["test"]["calibrated"]["roc_auc"] or float("-inf"),
    )[0]
    best_brier_config = min(
        configurations.items(),
        key=lambda item: item[1]["metrics"]["test"]["calibrated"]["brier_score"] or float("inf"),
    )[0]
    promising_configs = [
        name
        for name, payload in configurations.items()
        if payload["comparison_vs_frozen_reduced"]["delta_vs_frozen_reduced"]["test_roc_auc"] > 0.0
        and payload["comparison_vs_frozen_reduced"]["delta_vs_frozen_reduced"]["test_pr_auc"] > 0.0
        and payload["comparison_vs_frozen_reduced"]["delta_vs_frozen_reduced"]["calibrated_test_brier"] < 0.0
    ]

    if not promising_configs:
        recommendation = "archive all meta-scores"
        rationale = "No ablation configuration beats frozen REDUCED simultaneously on calibrated test ROC-AUC, calibrated test PR-AUC, and calibrated test Brier."
    elif best_roc_config == "REDUCED_PLUS_WALLET":
        recommendation = "keep wallet only"
        rationale = "Wallet-only is the strongest held-out configuration and clears the frozen REDUCED baseline on calibrated test metrics."
    elif best_roc_config == "REDUCED_PLUS_TELCO":
        recommendation = "keep telco only"
        rationale = "Telco-only is the strongest held-out configuration and clears the frozen REDUCED baseline on calibrated test metrics."
    elif best_roc_config == "REDUCED_PLUS_TELCO_WALLET":
        recommendation = "keep wallet+telco"
        rationale = "The joint telco+wallet configuration is the strongest held-out configuration and clears the frozen REDUCED baseline on calibrated test metrics."
    else:
        recommendation = "continue to LGBM master only if one ablation is promising"
        rationale = "At least one ablation is promising, but the strongest candidate is not one of the simpler keep-only options."

    return {
        "best_by_calibrated_test_roc_auc": best_roc_config,
        "best_by_calibrated_test_brier": best_brier_config,
        "promising_configurations": promising_configs,
        "recommendation": recommendation,
        "rationale": rationale,
    }


def _format_metric(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


def _build_ablation_summary_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# ALT_STACKED_REDUCED Ablation Summary",
        "",
        "| Configuration | Meta Columns | Val ROC | Val PR | Test Raw ROC | Test Raw PR | Test Cal ROC | Test Cal PR | Test Cal Brier | Delta Val ROC | Delta Test ROC | Delta Test PR | Delta Cal Brier |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for config_name, payload in report["configurations"].items():
        metrics = payload["metrics"]
        deltas = payload["comparison_vs_frozen_reduced"]["delta_vs_frozen_reduced"]
        meta_columns = payload["feature_summary"]["selected_meta_columns"]
        lines.append(
            "| "
            + " | ".join(
                [
                    config_name,
                    ", ".join(meta_columns) if meta_columns else "None",
                    _format_metric(metrics["val_model"]["raw"]["roc_auc"]),
                    _format_metric(metrics["val_model"]["raw"]["pr_auc"]),
                    _format_metric(metrics["test"]["raw"]["roc_auc"]),
                    _format_metric(metrics["test"]["raw"]["pr_auc"]),
                    _format_metric(metrics["test"]["calibrated"]["roc_auc"]),
                    _format_metric(metrics["test"]["calibrated"]["pr_auc"]),
                    _format_metric(metrics["test"]["calibrated"]["brier_score"]),
                    _format_metric(deltas["val_model_roc_auc"]),
                    _format_metric(deltas["test_roc_auc"]),
                    _format_metric(deltas["test_pr_auc"]),
                    _format_metric(deltas["calibrated_test_brier"]),
                ]
            )
            + " |"
        )
    recommendation = report["recommendation"]
    lines.extend(
        [
            "",
            f"Best calibrated test ROC-AUC: `{recommendation['best_by_calibrated_test_roc_auc']}`",
            f"Best calibrated test Brier: `{recommendation['best_by_calibrated_test_brier']}`",
            f"Recommendation: `{recommendation['recommendation']}`",
            "",
            recommendation["rationale"],
        ]
    )
    return "\n".join(lines) + "\n"


def load_frozen_reduced_baseline_metrics(
    bundle: DatasetBundle,
    *,
    artifact_dir: str = ARTIFACT_DIR,
    processed_dir: str = DATA_DIR,
) -> dict[str, Any]:
    runtime = load_artifacts(
        artifact_dir=artifact_dir,
        processed_dir=processed_dir,
        strict_artifacts=True,
    )
    reduced_builder = runtime["reduced_builder"]
    reduced_model = runtime["reduced_model"]
    reduced_calibrator = runtime["reduced_calibrator"]

    val_model_X = build_reduced(bundle.val_model, reduced_builder)
    val_policy_X = build_reduced(bundle.val_policy, reduced_builder)
    test_X = build_reduced(bundle.test, reduced_builder)

    val_model_y = bundle.val_model["TARGET"].to_numpy(dtype=int)
    val_policy_y = bundle.val_policy["TARGET"].to_numpy(dtype=int)
    test_y = bundle.test["TARGET"].to_numpy(dtype=int)

    val_model_raw_scores = np.asarray(reduced_model.predict_proba(val_model_X), dtype=float)[:, 1]
    val_policy_raw_scores = np.asarray(reduced_model.predict_proba(val_policy_X), dtype=float)[:, 1]
    test_raw_scores = np.asarray(reduced_model.predict_proba(test_X), dtype=float)[:, 1]
    val_model_calibrated_scores = np.asarray(reduced_calibrator.predict(val_model_raw_scores), dtype=float)
    val_policy_calibrated_scores = np.asarray(reduced_calibrator.predict(val_policy_raw_scores), dtype=float)
    test_calibrated_scores = np.asarray(reduced_calibrator.predict(test_raw_scores), dtype=float)

    return {
        "source": "frozen_runtime_reduced_artifacts",
        "val_model": evaluate_alt_stacked_reduced_master(
            val_model_y,
            val_model_raw_scores,
            val_model_calibrated_scores,
        ),
        "val_policy": evaluate_alt_stacked_reduced_master(
            val_policy_y,
            val_policy_raw_scores,
            val_policy_calibrated_scores,
        ),
        "test": evaluate_alt_stacked_reduced_master(
            test_y,
            test_raw_scores,
            test_calibrated_scores,
        ),
    }


def _build_submodel_report(
    *,
    experiment_name: str,
    selected_candidate: str,
    selected_params: dict[str, Any],
    feature_columns: list[str],
    bundle: DatasetBundle,
    alignment_summary: dict[str, dict[str, Any]],
    n_oof_folds: int,
    oof_metadata: dict[str, Any],
    val_model_y: np.ndarray,
    val_model_predictions: pd.DataFrame,
    val_policy_y: np.ndarray,
    val_policy_predictions: pd.DataFrame,
    test_y: np.ndarray,
    test_predictions: pd.DataFrame,
    score_column: str,
    model_path: str,
    prediction_paths: dict[str, str],
    report_path: str,
    sidecar_label: str,
) -> dict[str, Any]:
    return {
        "experiment_name": experiment_name,
        "offline_only": True,
        "runtime_artifacts_unchanged": True,
        "selected_model_family": "xgboost",
        "selected_candidate": selected_candidate,
        "selected_params": selected_params,
        "feature_columns": feature_columns,
        "sample_counts": {
            "train": int(len(bundle.train)),
            "val_model": int(len(bundle.val_model)),
            "val_policy": int(len(bundle.val_policy)),
            "test": int(len(bundle.test)),
        },
        "alignment_summary": alignment_summary,
        "oof_strategy": {
            "type": "blocked_forward_chaining",
            "n_folds": int(n_oof_folds),
            "uses_train_rows_only": True,
            "future_rows_never_used_for_earlier_fold_fits": True,
            "notes": [
                "Train OOF predictions are produced from contiguous order-respecting validation blocks.",
                "Each fold fits only on rows strictly earlier than its validation block.",
                "When a fold has no usable history or only one class in history, a constant prior fallback is used instead of an in-sample model score.",
            ],
        },
        "metrics": {
            "train_oof": oof_metadata["metrics"],
            "val_model": _collect_submodel_metrics(
                val_model_y,
                val_model_predictions[score_column].to_numpy(dtype=float),
            ),
            "val_policy": _collect_submodel_metrics(
                val_policy_y,
                val_policy_predictions[score_column].to_numpy(dtype=float),
            ),
            "test": _collect_submodel_metrics(
                test_y,
                test_predictions[score_column].to_numpy(dtype=float),
            ),
        },
        "artifacts": {
            "model": model_path,
            "train_oof_predictions": prediction_paths["train_oof"],
            "val_model_predictions": prediction_paths["val_model"],
            "val_policy_predictions": prediction_paths["val_policy"],
            "test_predictions": prediction_paths["test"],
            "report": report_path,
        },
        "fold_summary": oof_metadata["folds"],
        "notes": [
            f"{experiment_name} is an offline-only experiment and does not overwrite runtime REDUCED artifacts.",
            f"{experiment_name} is trained only on {sidecar_label} sidecar features aligned by SK_ID_CURR.",
            "val_model is used for candidate selection, val_policy scores are reserved for later calibration work, and test is final confirmation only.",
        ],
    }


def run_telco_submodel_experiment(
    *,
    telco_sidecar_path: str | Path,
    artifact_dir: str = ARTIFACT_DIR,
    processed_dir: str = DATA_DIR,
    raw_dir: str = "data/raw/",
    n_oof_folds: int = DEFAULT_SUBMODEL_OOF_FOLDS,
) -> dict[str, Any]:
    experiment_artifact_dir = _alt_artifact_dir(artifact_dir)
    os.makedirs(experiment_artifact_dir, exist_ok=True)

    bundle = load_alt_stacked_reduced_bundle(processed_dir=processed_dir, raw_dir=raw_dir)
    aligned_frames, alignment_summary, telco_sidecar_df = load_and_align_telco_sidecar_to_bundle(
        bundle,
        telco_sidecar_path,
    )
    telco_feature_columns = extract_telco_feature_columns(telco_sidecar_df)

    train_X = aligned_frames["train"][telco_feature_columns].reset_index(drop=True)
    val_model_X = aligned_frames["val_model"][telco_feature_columns].reset_index(drop=True)
    val_policy_X = aligned_frames["val_policy"][telco_feature_columns].reset_index(drop=True)
    test_X = aligned_frames["test"][telco_feature_columns].reset_index(drop=True)

    train_y = bundle.train["TARGET"].to_numpy(dtype=int)
    val_model_y = bundle.val_model["TARGET"].to_numpy(dtype=int)
    val_policy_y = bundle.val_policy["TARGET"].to_numpy(dtype=int)
    test_y = bundle.test["TARGET"].to_numpy(dtype=int)

    fitted = fit_telco_submodel(train_X, train_y, val_model_X, val_model_y)
    model = fitted["model"]

    train_oof_predictions, oof_metadata = generate_telco_oof_predictions(
        bundle.train[SK_ID_CURR_COL],
        train_X,
        train_y,
        fitted["selected_params"],
        n_folds=n_oof_folds,
    )
    val_model_predictions = generate_telco_prediction_frame(
        bundle.val_model[SK_ID_CURR_COL],
        val_model_X,
        "val_model",
        model,
    )
    val_policy_predictions = generate_telco_prediction_frame(
        bundle.val_policy[SK_ID_CURR_COL],
        val_policy_X,
        "val_policy",
        model,
    )
    test_predictions = generate_telco_prediction_frame(
        bundle.test[SK_ID_CURR_COL],
        test_X,
        "test",
        model,
    )

    paths = _prediction_path_map(
        experiment_artifact_dir,
        train_oof_filename=TELCO_OOF_PREDICTIONS_FILENAME,
        val_model_filename=TELCO_VAL_MODEL_PREDICTIONS_FILENAME,
        val_policy_filename=TELCO_VAL_POLICY_PREDICTIONS_FILENAME,
        test_filename=TELCO_TEST_PREDICTIONS_FILENAME,
    )
    model_path = _artifact_path(experiment_artifact_dir, TELCO_SUBMODEL_FILENAME)
    report_path = _artifact_path(experiment_artifact_dir, TELCO_SUBMODEL_REPORT_FILENAME)

    _write_prediction_frame(paths["train_oof"], train_oof_predictions, score_column=TELCO_SCORE_COL)
    _write_prediction_frame(paths["val_model"], val_model_predictions, score_column=TELCO_SCORE_COL)
    _write_prediction_frame(paths["val_policy"], val_policy_predictions, score_column=TELCO_SCORE_COL)
    _write_prediction_frame(paths["test"], test_predictions, score_column=TELCO_SCORE_COL)
    joblib.dump(model, model_path)

    report = _build_submodel_report(
        experiment_name="TELCO_SUBMODEL",
        selected_candidate=fitted["selected_candidate"],
        selected_params=fitted["selected_params"],
        feature_columns=telco_feature_columns,
        bundle=bundle,
        alignment_summary=alignment_summary,
        n_oof_folds=n_oof_folds,
        oof_metadata=oof_metadata,
        val_model_y=val_model_y,
        val_model_predictions=val_model_predictions,
        val_policy_y=val_policy_y,
        val_policy_predictions=val_policy_predictions,
        test_y=test_y,
        test_predictions=test_predictions,
        score_column=TELCO_SCORE_COL,
        model_path=model_path,
        prediction_paths=paths,
        report_path=report_path,
        sidecar_label="telco",
    )
    _write_json(report_path, report)
    report["report_path"] = report_path
    return report


def run_wallet_submodel_experiment(
    *,
    wallet_sidecar_path: str | Path,
    artifact_dir: str = ARTIFACT_DIR,
    processed_dir: str = DATA_DIR,
    raw_dir: str = "data/raw/",
    n_oof_folds: int = DEFAULT_SUBMODEL_OOF_FOLDS,
) -> dict[str, Any]:
    experiment_artifact_dir = _alt_artifact_dir(artifact_dir)
    os.makedirs(experiment_artifact_dir, exist_ok=True)

    bundle = load_alt_stacked_reduced_bundle(processed_dir=processed_dir, raw_dir=raw_dir)
    aligned_frames, alignment_summary, wallet_sidecar_df = load_and_align_wallet_sidecar_to_bundle(
        bundle,
        wallet_sidecar_path,
    )
    wallet_feature_columns = extract_wallet_feature_columns(wallet_sidecar_df)

    train_X = aligned_frames["train"][wallet_feature_columns].reset_index(drop=True)
    val_model_X = aligned_frames["val_model"][wallet_feature_columns].reset_index(drop=True)
    val_policy_X = aligned_frames["val_policy"][wallet_feature_columns].reset_index(drop=True)
    test_X = aligned_frames["test"][wallet_feature_columns].reset_index(drop=True)

    train_y = bundle.train["TARGET"].to_numpy(dtype=int)
    val_model_y = bundle.val_model["TARGET"].to_numpy(dtype=int)
    val_policy_y = bundle.val_policy["TARGET"].to_numpy(dtype=int)
    test_y = bundle.test["TARGET"].to_numpy(dtype=int)

    fitted = fit_wallet_submodel(train_X, train_y, val_model_X, val_model_y)
    model = fitted["model"]

    train_oof_predictions, oof_metadata = generate_wallet_oof_predictions(
        bundle.train[SK_ID_CURR_COL],
        train_X,
        train_y,
        fitted["selected_params"],
        n_folds=n_oof_folds,
    )
    val_model_predictions = generate_wallet_prediction_frame(
        bundle.val_model[SK_ID_CURR_COL],
        val_model_X,
        "val_model",
        model,
    )
    val_policy_predictions = generate_wallet_prediction_frame(
        bundle.val_policy[SK_ID_CURR_COL],
        val_policy_X,
        "val_policy",
        model,
    )
    test_predictions = generate_wallet_prediction_frame(
        bundle.test[SK_ID_CURR_COL],
        test_X,
        "test",
        model,
    )

    paths = _prediction_path_map(
        experiment_artifact_dir,
        train_oof_filename=WALLET_OOF_PREDICTIONS_FILENAME,
        val_model_filename=WALLET_VAL_MODEL_PREDICTIONS_FILENAME,
        val_policy_filename=WALLET_VAL_POLICY_PREDICTIONS_FILENAME,
        test_filename=WALLET_TEST_PREDICTIONS_FILENAME,
    )
    model_path = _artifact_path(experiment_artifact_dir, WALLET_SUBMODEL_FILENAME)
    report_path = _artifact_path(experiment_artifact_dir, WALLET_SUBMODEL_REPORT_FILENAME)

    _write_prediction_frame(paths["train_oof"], train_oof_predictions, score_column=WALLET_SCORE_COL)
    _write_prediction_frame(paths["val_model"], val_model_predictions, score_column=WALLET_SCORE_COL)
    _write_prediction_frame(paths["val_policy"], val_policy_predictions, score_column=WALLET_SCORE_COL)
    _write_prediction_frame(paths["test"], test_predictions, score_column=WALLET_SCORE_COL)
    joblib.dump(model, model_path)

    report = _build_submodel_report(
        experiment_name="WALLET_SUBMODEL",
        selected_candidate=fitted["selected_candidate"],
        selected_params=fitted["selected_params"],
        feature_columns=wallet_feature_columns,
        bundle=bundle,
        alignment_summary=alignment_summary,
        n_oof_folds=n_oof_folds,
        oof_metadata=oof_metadata,
        val_model_y=val_model_y,
        val_model_predictions=val_model_predictions,
        val_policy_y=val_policy_y,
        val_policy_predictions=val_policy_predictions,
        test_y=test_y,
        test_predictions=test_predictions,
        score_column=WALLET_SCORE_COL,
        model_path=model_path,
        prediction_paths=paths,
        report_path=report_path,
        sidecar_label="wallet",
    )
    _write_json(report_path, report)
    report["report_path"] = report_path
    return report


def run_ecommerce_submodel_experiment(
    *,
    ecommerce_sidecar_path: str | Path,
    artifact_dir: str = ARTIFACT_DIR,
    processed_dir: str = DATA_DIR,
    raw_dir: str = "data/raw/",
    n_oof_folds: int = DEFAULT_SUBMODEL_OOF_FOLDS,
) -> dict[str, Any]:
    experiment_artifact_dir = _alt_artifact_dir(artifact_dir)
    os.makedirs(experiment_artifact_dir, exist_ok=True)

    bundle = load_alt_stacked_reduced_bundle(processed_dir=processed_dir, raw_dir=raw_dir)
    aligned_frames, alignment_summary, ecommerce_sidecar_df = load_and_align_ecommerce_sidecar_to_bundle(
        bundle,
        ecommerce_sidecar_path,
    )
    ecommerce_feature_columns = extract_ecommerce_feature_columns(ecommerce_sidecar_df)

    train_X = aligned_frames["train"][ecommerce_feature_columns].reset_index(drop=True)
    val_model_X = aligned_frames["val_model"][ecommerce_feature_columns].reset_index(drop=True)
    val_policy_X = aligned_frames["val_policy"][ecommerce_feature_columns].reset_index(drop=True)
    test_X = aligned_frames["test"][ecommerce_feature_columns].reset_index(drop=True)

    train_y = bundle.train["TARGET"].to_numpy(dtype=int)
    val_model_y = bundle.val_model["TARGET"].to_numpy(dtype=int)
    val_policy_y = bundle.val_policy["TARGET"].to_numpy(dtype=int)
    test_y = bundle.test["TARGET"].to_numpy(dtype=int)

    fitted = fit_ecommerce_submodel(train_X, train_y, val_model_X, val_model_y)
    model = fitted["model"]

    train_oof_predictions, oof_metadata = generate_ecommerce_oof_predictions(
        bundle.train[SK_ID_CURR_COL],
        train_X,
        train_y,
        fitted["selected_params"],
        n_folds=n_oof_folds,
    )
    val_model_predictions = generate_ecommerce_prediction_frame(
        bundle.val_model[SK_ID_CURR_COL],
        val_model_X,
        "val_model",
        model,
    )
    val_policy_predictions = generate_ecommerce_prediction_frame(
        bundle.val_policy[SK_ID_CURR_COL],
        val_policy_X,
        "val_policy",
        model,
    )
    test_predictions = generate_ecommerce_prediction_frame(
        bundle.test[SK_ID_CURR_COL],
        test_X,
        "test",
        model,
    )

    paths = _prediction_path_map(
        experiment_artifact_dir,
        train_oof_filename=ECOMMERCE_OOF_PREDICTIONS_FILENAME,
        val_model_filename=ECOMMERCE_VAL_MODEL_PREDICTIONS_FILENAME,
        val_policy_filename=ECOMMERCE_VAL_POLICY_PREDICTIONS_FILENAME,
        test_filename=ECOMMERCE_TEST_PREDICTIONS_FILENAME,
    )
    model_path = _artifact_path(experiment_artifact_dir, ECOMMERCE_SUBMODEL_FILENAME)
    report_path = _artifact_path(experiment_artifact_dir, ECOMMERCE_SUBMODEL_REPORT_FILENAME)

    _write_prediction_frame(paths["train_oof"], train_oof_predictions, score_column=ECOMMERCE_SCORE_COL)
    _write_prediction_frame(paths["val_model"], val_model_predictions, score_column=ECOMMERCE_SCORE_COL)
    _write_prediction_frame(paths["val_policy"], val_policy_predictions, score_column=ECOMMERCE_SCORE_COL)
    _write_prediction_frame(paths["test"], test_predictions, score_column=ECOMMERCE_SCORE_COL)
    joblib.dump(model, model_path)

    report = _build_submodel_report(
        experiment_name="ECOMMERCE_SUBMODEL",
        selected_candidate=fitted["selected_candidate"],
        selected_params=fitted["selected_params"],
        feature_columns=ecommerce_feature_columns,
        bundle=bundle,
        alignment_summary=alignment_summary,
        n_oof_folds=n_oof_folds,
        oof_metadata=oof_metadata,
        val_model_y=val_model_y,
        val_model_predictions=val_model_predictions,
        val_policy_y=val_policy_y,
        val_policy_predictions=val_policy_predictions,
        test_y=test_y,
        test_predictions=test_predictions,
        score_column=ECOMMERCE_SCORE_COL,
        model_path=model_path,
        prediction_paths=paths,
        report_path=report_path,
        sidecar_label="ecommerce",
    )
    _write_json(report_path, report)
    report["report_path"] = report_path
    return report


def run_meta_score_assembly_experiment(
    *,
    artifact_dir: str = ARTIFACT_DIR,
    processed_dir: str = DATA_DIR,
    raw_dir: str = "data/raw/",
) -> dict[str, Any]:
    experiment_artifact_dir = _alt_artifact_dir(artifact_dir)
    stacked_processed_dir = _stacked_processed_dir(processed_dir)
    os.makedirs(experiment_artifact_dir, exist_ok=True)
    os.makedirs(stacked_processed_dir, exist_ok=True)

    bundle = load_alt_stacked_reduced_bundle(processed_dir=processed_dir, raw_dir=raw_dir)
    telco_tables = load_telco_prediction_tables(bundle, artifact_dir=artifact_dir)
    wallet_tables = load_wallet_prediction_tables(bundle, artifact_dir=artifact_dir)
    ecommerce_tables = load_ecommerce_prediction_tables(bundle, artifact_dir=artifact_dir)

    meta_score_frames = build_all_meta_score_frames(bundle, telco_tables, wallet_tables, ecommerce_tables)
    stacked = build_stacked_reduced_split_matrices(bundle, meta_score_frames, processed_dir=processed_dir)
    canonical_matrices: dict[str, pd.DataFrame] = stacked["canonical_matrices"]
    stacked_matrices: dict[str, pd.DataFrame] = stacked["stacked_matrices"]

    meta_score_paths = _meta_score_artifact_path_map(experiment_artifact_dir)
    stacked_matrix_paths = _stacked_matrix_path_map(processed_dir)
    for split_name, frame in meta_score_frames.items():
        frame.to_csv(meta_score_paths[split_name], index=False)
    for split_name, matrix in stacked_matrices.items():
        matrix.to_pickle(stacked_matrix_paths[split_name])

    prediction_specs = _side_prediction_specs(artifact_dir)
    lineage = {
        "experiment_name": "ALT_STACKED_REDUCED_META_ASSEMBLY",
        "offline_only": True,
        "runtime_artifacts_unchanged": True,
        "meta_columns": [META_SCORE_TELCO_COL, META_SCORE_WALLET_COL, META_SCORE_ECOMMERCE_COL],
        "split_sources": {
            split_name: {
                "telco": {
                    "path": prediction_specs["telco"]["paths"]["train_oof" if split_name == "train" else split_name],
                    "score_column": TELCO_SCORE_COL,
                    "expected_split_name": "train_oof" if split_name == "train" else split_name,
                    "meta_column": META_SCORE_TELCO_COL,
                },
                "wallet": {
                    "path": prediction_specs["wallet"]["paths"]["train_oof" if split_name == "train" else split_name],
                    "score_column": WALLET_SCORE_COL,
                    "expected_split_name": "train_oof" if split_name == "train" else split_name,
                    "meta_column": META_SCORE_WALLET_COL,
                },
                "ecommerce": {
                    "path": prediction_specs["ecommerce"]["paths"]["train_oof" if split_name == "train" else split_name],
                    "score_column": ECOMMERCE_SCORE_COL,
                    "expected_split_name": "train_oof" if split_name == "train" else split_name,
                    "meta_column": META_SCORE_ECOMMERCE_COL,
                },
            }
            for split_name in ("train", "val_model", "val_policy", "test")
        },
        "meta_score_artifacts": meta_score_paths,
        "stacked_matrix_artifacts": stacked_matrix_paths,
    }
    lineage_path = _artifact_path(experiment_artifact_dir, META_SCORE_LINEAGE_FILENAME)
    _write_json(lineage_path, lineage)

    summary = {
        "experiment_name": "ALT_STACKED_REDUCED_META_ASSEMBLY",
        "offline_only": True,
        "runtime_artifacts_unchanged": True,
        "meta_columns": [META_SCORE_TELCO_COL, META_SCORE_WALLET_COL, META_SCORE_ECOMMERCE_COL],
        "split_summary": {
            split_name: {
                "row_count": int(len(meta_score_frames[split_name])),
                "meta_score_path": meta_score_paths[split_name],
                "stacked_matrix_path": stacked_matrix_paths[split_name],
                "canonical_reduced_feature_count": int(canonical_matrices[split_name].shape[1]),
                "stacked_reduced_feature_count": int(stacked_matrices[split_name].shape[1]),
                "appended_meta_feature_count": int(stacked_matrices[split_name].shape[1] - canonical_matrices[split_name].shape[1]),
                "ids_match_bundle": meta_score_frames[split_name][SK_ID_CURR_COL].tolist()
                == getattr(bundle, split_name)[SK_ID_CURR_COL].tolist(),
            }
            for split_name in ("train", "val_model", "val_policy", "test")
        },
        "meta_score_artifacts": meta_score_paths,
        "stacked_matrix_artifacts": stacked_matrix_paths,
        "lineage_artifact": lineage_path,
    }
    summary_path = _artifact_path(experiment_artifact_dir, STACKED_REDUCED_SPLIT_SUMMARY_FILENAME)
    _write_json(summary_path, summary)
    summary["summary_path"] = summary_path
    summary["lineage_path"] = lineage_path
    return summary


def run_alt_stacked_reduced_master_xgb_experiment(
    *,
    artifact_dir: str = ARTIFACT_DIR,
    processed_dir: str = DATA_DIR,
    raw_dir: str = "data/raw/",
) -> dict[str, Any]:
    experiment_artifact_dir = _alt_artifact_dir(artifact_dir)
    os.makedirs(experiment_artifact_dir, exist_ok=True)

    bundle = load_alt_stacked_reduced_bundle(processed_dir=processed_dir, raw_dir=raw_dir)
    stacked_matrices = load_stacked_reduced_split_matrices(processed_dir=processed_dir)

    train_ids, train_X, train_y = extract_X_y_from_stacked_split(bundle, stacked_matrices["train"], "train")
    val_model_ids, val_model_X, val_model_y = extract_X_y_from_stacked_split(bundle, stacked_matrices["val_model"], "val_model")
    val_policy_ids, val_policy_X, val_policy_y = extract_X_y_from_stacked_split(bundle, stacked_matrices["val_policy"], "val_policy")
    test_ids, test_X, test_y = extract_X_y_from_stacked_split(bundle, stacked_matrices["test"], "test")

    for required_meta_col in [META_SCORE_TELCO_COL, META_SCORE_WALLET_COL, META_SCORE_ECOMMERCE_COL]:
        if required_meta_col not in train_X.columns:
            raise ValueError(f"Stacked train matrix is missing required meta column: {required_meta_col}")

    fitted = fit_alt_stacked_reduced_master_xgb(train_X, train_y, val_model_X, val_model_y)
    model = fitted["model"]

    val_model_raw_scores = np.asarray(fitted["val_model_scores"], dtype=float)
    val_policy_raw_scores = np.asarray(model.predict_proba(val_policy_X), dtype=float)[:, 1]
    calibration = calibrate_alt_stacked_reduced_master(val_policy_y, val_policy_raw_scores)
    calibrator = calibration["calibrator"]
    val_model_calibrated_scores = np.asarray(calibrator.predict(val_model_raw_scores), dtype=float)
    test_raw_scores = np.asarray(model.predict_proba(test_X), dtype=float)[:, 1]
    test_calibrated_scores = np.asarray(calibrator.predict(test_raw_scores), dtype=float)

    val_model_predictions = _build_master_prediction_frame(
        val_model_ids,
        "val_model",
        val_model_raw_scores,
        val_model_calibrated_scores,
    )
    val_policy_predictions = _build_master_prediction_frame(
        val_policy_ids,
        "val_policy",
        val_policy_raw_scores,
        calibration["val_policy_calibrated_scores"],
    )
    test_predictions = _build_master_prediction_frame(
        test_ids,
        "test",
        test_raw_scores,
        test_calibrated_scores,
    )

    prediction_paths = _master_prediction_path_map(experiment_artifact_dir)
    model_path = _artifact_path(experiment_artifact_dir, MASTER_XGB_MODEL_FILENAME)
    calibrator_path = _artifact_path(experiment_artifact_dir, MASTER_XGB_CALIBRATOR_FILENAME)
    report_path = _artifact_path(experiment_artifact_dir, MASTER_XGB_REPORT_FILENAME)

    _write_master_prediction_frame(prediction_paths["val_model"], val_model_predictions)
    _write_master_prediction_frame(prediction_paths["val_policy"], val_policy_predictions)
    _write_master_prediction_frame(prediction_paths["test"], test_predictions)
    joblib.dump(model, model_path)
    joblib.dump(calibrator, calibrator_path)

    frozen_baseline = load_frozen_reduced_baseline_metrics(
        bundle,
        artifact_dir=artifact_dir,
        processed_dir=processed_dir,
    )
    val_model_metrics = evaluate_alt_stacked_reduced_master(
        val_model_y,
        val_model_raw_scores,
        val_model_calibrated_scores,
    )
    val_policy_metrics = evaluate_alt_stacked_reduced_master(
        val_policy_y,
        val_policy_raw_scores,
        calibration["val_policy_calibrated_scores"],
    )
    test_metrics = evaluate_alt_stacked_reduced_master(
        test_y,
        test_raw_scores,
        test_calibrated_scores,
    )

    report = {
        "experiment_name": "ALT_STACKED_REDUCED_MASTER_XGB",
        "offline_only": True,
        "runtime_artifacts_unchanged": True,
        "selected_model_family": "xgboost",
        "selected_candidate": fitted["selected_candidate"],
        "selected_params": fitted["selected_params"],
        "selected_calibrator": calibration["selected_calibrator"],
        "sample_counts": {
            "train": int(len(train_X)),
            "val_model": int(len(val_model_X)),
            "val_policy": int(len(val_policy_X)),
            "test": int(len(test_X)),
        },
        "feature_summary": {
            "train_feature_count": int(train_X.shape[1]),
            "meta_feature_columns": [META_SCORE_TELCO_COL, META_SCORE_WALLET_COL, META_SCORE_ECOMMERCE_COL],
            "contains_meta_columns": {
                META_SCORE_TELCO_COL: META_SCORE_TELCO_COL in train_X.columns,
                META_SCORE_WALLET_COL: META_SCORE_WALLET_COL in train_X.columns,
                META_SCORE_ECOMMERCE_COL: META_SCORE_ECOMMERCE_COL in train_X.columns,
            },
        },
        "metrics": {
            "val_model": val_model_metrics,
            "val_policy": val_policy_metrics,
            "test": test_metrics,
        },
        "comparison_vs_frozen_reduced": _comparison_vs_frozen_reduced(
            frozen_baseline=frozen_baseline,
            val_model_metrics=val_model_metrics,
            test_metrics=test_metrics,
        ),
        "artifacts": {
            "model": model_path,
            "calibrator": calibrator_path,
            "val_model_predictions": prediction_paths["val_model"],
            "val_policy_predictions": prediction_paths["val_policy"],
            "test_predictions": prediction_paths["test"],
            "report": report_path,
        },
        "notes": [
            "ALT_STACKED_REDUCED_MASTER_XGB is an offline-only stacked master candidate and does not overwrite frozen runtime REDUCED artifacts.",
            "The master model is fit on train, selected on val_model, calibrated on val_policy, and confirmed once on test.",
            "The stacked inputs preserve all canonical REDUCED features and append META_SCORE_TELCO, META_SCORE_WALLET, and META_SCORE_ECOMMERCE as additional columns only.",
            "Comparison deltas use raw val_model metrics and calibrated test metrics versus the frozen REDUCED runtime baseline.",
        ],
    }
    _write_json(report_path, report)
    report["report_path"] = report_path
    return report


def run_alt_stacked_reduced_ablation_experiment(
    *,
    artifact_dir: str = ARTIFACT_DIR,
    processed_dir: str = DATA_DIR,
    raw_dir: str = "data/raw/",
) -> dict[str, Any]:
    experiment_artifact_dir = _alt_artifact_dir(artifact_dir)
    os.makedirs(experiment_artifact_dir, exist_ok=True)

    bundle = load_alt_stacked_reduced_bundle(processed_dir=processed_dir, raw_dir=raw_dir)
    stored_stacked_matrices = load_stacked_reduced_split_matrices(processed_dir=processed_dir)
    frozen_baseline = load_frozen_reduced_baseline_metrics(
        bundle,
        artifact_dir=artifact_dir,
        processed_dir=processed_dir,
    )

    configurations: dict[str, dict[str, Any]] = {}
    for config_name, selected_meta_columns in ABLATION_CONFIG_META_COLUMNS.items():
        config_matrices = build_ablation_stacked_matrices_for_config(
            stored_stacked_matrices,
            selected_meta_columns,
        )
        train_ids, train_X, train_y = extract_X_y_from_stacked_split(bundle, config_matrices["train"], "train")
        val_model_ids, val_model_X, val_model_y = extract_X_y_from_stacked_split(bundle, config_matrices["val_model"], "val_model")
        val_policy_ids, val_policy_X, val_policy_y = extract_X_y_from_stacked_split(bundle, config_matrices["val_policy"], "val_policy")
        test_ids, test_X, test_y = extract_X_y_from_stacked_split(bundle, config_matrices["test"], "test")

        fitted = fit_alt_stacked_reduced_master_xgb(train_X, train_y, val_model_X, val_model_y)
        model = fitted["model"]

        val_model_raw_scores = np.asarray(fitted["val_model_scores"], dtype=float)
        val_policy_raw_scores = np.asarray(model.predict_proba(val_policy_X), dtype=float)[:, 1]
        calibration = calibrate_alt_stacked_reduced_master(val_policy_y, val_policy_raw_scores)
        val_model_calibrated_scores = np.asarray(calibration["calibrator"].predict(val_model_raw_scores), dtype=float)
        test_raw_scores = np.asarray(model.predict_proba(test_X), dtype=float)[:, 1]
        test_calibrated_scores = np.asarray(calibration["calibrator"].predict(test_raw_scores), dtype=float)

        val_model_metrics = evaluate_alt_stacked_reduced_master(
            val_model_y,
            val_model_raw_scores,
            val_model_calibrated_scores,
        )
        val_policy_metrics = evaluate_alt_stacked_reduced_master(
            val_policy_y,
            val_policy_raw_scores,
            calibration["val_policy_calibrated_scores"],
        )
        test_metrics = evaluate_alt_stacked_reduced_master(
            test_y,
            test_raw_scores,
            test_calibrated_scores,
        )

        configurations[config_name] = {
            "selected_model_family": "xgboost",
            "selected_candidate": fitted["selected_candidate"],
            "selected_params": fitted["selected_params"],
            "selected_calibrator": calibration["selected_calibrator"],
            "sample_counts": {
                "train": int(len(train_ids)),
                "val_model": int(len(val_model_ids)),
                "val_policy": int(len(val_policy_ids)),
                "test": int(len(test_ids)),
            },
            "feature_summary": {
                "canonical_feature_count": int(len([column for column in train_X.columns if column not in ALL_META_SCORE_COLS])),
                "train_feature_count": int(train_X.shape[1]),
                "selected_meta_columns": selected_meta_columns,
                "contains_meta_columns": {
                    meta_col: meta_col in train_X.columns
                    for meta_col in ALL_META_SCORE_COLS
                },
            },
            "row_count_checks": {
                split_name: int(len(config_matrices[split_name])) == int(len(stored_stacked_matrices[split_name]))
                for split_name in ("train", "val_model", "val_policy", "test")
            },
            "metrics": {
                "val_model": val_model_metrics,
                "val_policy": val_policy_metrics,
                "test": test_metrics,
            },
            "comparison_vs_frozen_reduced": _comparison_vs_frozen_reduced(
                frozen_baseline=frozen_baseline,
                val_model_metrics=val_model_metrics,
                test_metrics=test_metrics,
            ),
        }

    recommendation = _summarize_ablation_recommendation(configurations)
    report = {
        "experiment_name": "ALT_STACKED_REDUCED_ABLATION",
        "offline_only": True,
        "runtime_artifacts_unchanged": True,
        "source_stacked_matrix_artifacts": _stacked_matrix_path_map(processed_dir),
        "frozen_reduced_baseline": {
            "val_model_raw_roc_auc": frozen_baseline["val_model"]["raw"]["roc_auc"],
            "test_calibrated_roc_auc": frozen_baseline["test"]["calibrated"]["roc_auc"],
            "test_calibrated_pr_auc": frozen_baseline["test"]["calibrated"]["pr_auc"],
            "test_calibrated_brier": frozen_baseline["test"]["calibrated"]["brier_score"],
            "source": frozen_baseline["source"],
        },
        "configurations": configurations,
        "recommendation": recommendation,
        "artifacts": {
            "report": _artifact_path(experiment_artifact_dir, ABLATION_REPORT_FILENAME),
            "summary_markdown": _artifact_path(experiment_artifact_dir, ABLATION_SUMMARY_FILENAME),
        },
        "notes": [
            "Ablations reuse the already-built stacked REDUCED matrices and do not retrain side models.",
            "Each configuration is fit on train, compared on val_model, calibrated on val_policy, and confirmed once on test.",
            "BASE_REDUCED_ONLY reuses the canonical post-build_reduced feature set with all META_SCORE columns removed.",
        ],
    }
    report_path = report["artifacts"]["report"]
    summary_path = report["artifacts"]["summary_markdown"]
    _write_json(report_path, report)
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write(_build_ablation_summary_markdown(report))
    report["report_path"] = report_path
    report["summary_path"] = summary_path
    return report
