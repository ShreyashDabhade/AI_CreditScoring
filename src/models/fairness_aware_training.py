import json
import tempfile
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from configs.config import ARTIFACT_DIR, DATA_DIR
from src.fairness_audit import FAIRNESS_COLS, compute_fairness_metrics, derive_fairness_groups
from src.feature_engineering import build_full
from src.models.runtime_support import WeightedBlendModel
from src.models.train import (
    DEFAULT_FULL_FEATURE_VIEW,
    _artifact_path,
    _build_features,
    _candidate_lgbm_params,
    _candidate_model_params,
    _collect_metrics,
    _evaluate_calibrated_scores,
    _has_real_training_inputs,
    _load_real_bundle,
    _make_lgbm_model,
    _make_model,
    _predict_lightgbm_raw_pd,
    _processed_manifest_fingerprint,
    _select_weighted_average_blend,
    load_artifacts,
)

FAIRNESS_AWARE_MODELING_REPORT_FILENAME = "fairness_aware_modeling_experiment_report.json"
REGION_BALANCE_SHRINK = 0.50
REGION3_DEFAULT_BOOST = 1.50
APPROVE_THRESHOLD = 0.15
DECLINE_THRESHOLD = 0.35


@dataclass(frozen=True)
class FairnessInterventionSpec:
    name: str
    description: str
    weighting_strategy: str


def _predict_raw_pd(model: Any, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_raw_pd"):
        raw = model.predict_raw_pd(X)
    else:
        raw = model.predict_proba(X)[:, 1]
    return np.asarray(raw, dtype=float).reshape(-1)


def _prepare_group_frame(train_df: pd.DataFrame, split_df: pd.DataFrame) -> pd.DataFrame:
    fairness_cols_present = [col for col in FAIRNESS_COLS if col in split_df.columns]
    frame = split_df[["TARGET"] + fairness_cols_present].copy()
    q1 = float(train_df["AMT_INCOME_TOTAL"].quantile(1 / 3))
    q2 = float(train_df["AMT_INCOME_TOTAL"].quantile(2 / 3))
    return derive_fairness_groups(frame, q1, q2)


def _jsonable_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    records = df.replace({np.nan: None}).to_dict(orient="records")
    return [
        {
            key: (value.item() if hasattr(value, "item") else value)
            for key, value in record.items()
        }
        for record in records
    ]


def _strategy_specs() -> list[FairnessInterventionSpec]:
    return [
        FairnessInterventionSpec(
            name="region_balanced_mild",
            description=(
                "Half-strength region-balance weighting using shrunk inverse-support "
                "weights so minority regions contribute more without fully equalizing groups."
            ),
            weighting_strategy="region_balanced_mild",
        ),
        FairnessInterventionSpec(
            name="region3_default_boost",
            description=(
                "Targeted weight boost for REGION_3 default cases to reduce underprediction "
                "pressure in the worst primary-group region."
            ),
            weighting_strategy="region3_default_boost",
        ),
    ]


def _normalize_sample_weights(weights: np.ndarray) -> np.ndarray:
    arr = np.asarray(weights, dtype=float).reshape(-1)
    mean_weight = float(arr.mean())
    if mean_weight <= 0.0:
        raise ValueError("Sample weights must have positive mean")
    return arr / mean_weight


def _region_balance_weights(train_groups: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    region_counts = train_groups["FAIR_GROUP_PRIMARY"].value_counts().sort_index()
    target_count = float(region_counts.mean())
    raw_group_weights = {
        group: float(np.sqrt(target_count / count))
        for group, count in region_counts.items()
    }
    shrunk_group_weights = {
        group: float(1.0 + REGION_BALANCE_SHRINK * (weight - 1.0))
        for group, weight in raw_group_weights.items()
    }
    weights = train_groups["FAIR_GROUP_PRIMARY"].map(shrunk_group_weights).astype(float).to_numpy()
    normalized = _normalize_sample_weights(weights)
    effective_weights = (
        pd.DataFrame(
            {
                "region": train_groups["FAIR_GROUP_PRIMARY"].astype(str).to_numpy(),
                "weight": normalized,
            }
        )
        .groupby("region")["weight"]
        .mean()
        .to_dict()
    )
    details = {
        "strategy": "region_balanced_mild",
        "region_counts": {str(group): int(count) for group, count in region_counts.items()},
        "raw_group_weights": {str(group): float(weight) for group, weight in raw_group_weights.items()},
        "effective_group_weights": {str(group): float(weight) for group, weight in effective_weights.items()},
        "normalization": "sample weights normalized to mean 1.0 across train rows",
    }
    return normalized, details


def _region3_default_boost_weights(
    train_groups: pd.DataFrame,
    y_train: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    groups = train_groups["FAIR_GROUP_PRIMARY"].astype(str)
    target = np.asarray(y_train, dtype=int).reshape(-1)
    mask = (groups == "REGION_3").to_numpy() & (target == 1)
    weights = np.ones(len(train_groups), dtype=float)
    weights[mask] = REGION3_DEFAULT_BOOST
    normalized = _normalize_sample_weights(weights)
    details = {
        "strategy": "region3_default_boost",
        "boosted_group": "REGION_3",
        "boosted_target_value": 1,
        "boost_multiplier": float(REGION3_DEFAULT_BOOST),
        "boosted_rows": int(mask.sum()),
        "normalization": "sample weights normalized to mean 1.0 across train rows",
    }
    return normalized, details


def _build_train_weights(
    spec: FairnessInterventionSpec,
    train_groups: pd.DataFrame,
    y_train: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    if spec.weighting_strategy == "region_balanced_mild":
        return _region_balance_weights(train_groups)
    if spec.weighting_strategy == "region3_default_boost":
        return _region3_default_boost_weights(train_groups, y_train)
    raise ValueError(f"Unsupported weighting strategy: {spec.weighting_strategy}")


def _summarize_fairness_family(audit_df: pd.DataFrame, group_col: str) -> dict[str, Any]:
    metrics_df = compute_fairness_metrics(audit_df, group_col)
    evaluable = metrics_df[metrics_df["evaluable"] == True]  # noqa: E712
    summary = {
        "rows": int(len(metrics_df)),
        "evaluable_cells": int(len(evaluable)),
        "non_evaluable_cells": int(len(metrics_df) - len(evaluable)),
        "di_pass": False,
        "eod_pass": False,
        "brier_pass": False,
        "di_min": None,
        "eod_gap": None,
        "brier_ratio": None,
        "metrics_table": _jsonable_records(metrics_df),
    }
    if len(evaluable) > 0:
        summary.update({
            "di_pass": bool(evaluable["di_pass"].iloc[0]),
            "eod_pass": bool(evaluable["eod_pass"].iloc[0]),
            "brier_pass": bool(evaluable["brier_pass"].iloc[0]),
            "di_min": float(evaluable["di_ratio"].min()),
            "eod_gap": float(abs(evaluable["eod"].min())),
            "brier_ratio": float(evaluable["brier_ratio"].max()),
        })
    return summary


def _summarize_test_fairness(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    calibrated_pd: np.ndarray,
) -> dict[str, Any]:
    groups = _prepare_group_frame(train_df, test_df)
    audit_df = groups.copy()
    audit_df["CALIBRATED_PD"] = np.asarray(calibrated_pd, dtype=float)
    pd_values = audit_df["CALIBRATED_PD"].to_numpy()
    audit_df["DECISION"] = np.where(
        pd_values < APPROVE_THRESHOLD,
        "APPROVE",
        np.where(pd_values < DECLINE_THRESHOLD, "REVIEW", "DECLINE"),
    )
    primary_df = compute_fairness_metrics(audit_df, "FAIR_GROUP_PRIMARY")
    return {
        "decision_mix": audit_df["DECISION"].value_counts(normalize=True).sort_index().to_dict(),
        "families": {
            "PRIMARY": _summarize_fairness_family(audit_df, "FAIR_GROUP_PRIMARY"),
            "SECONDARY": _summarize_fairness_family(audit_df, "FAIR_GROUP_SECONDARY"),
            "TERTIARY": _summarize_fairness_family(audit_df, "FAIR_GROUP_TERTIARY"),
        },
        "primary_group_rows": _jsonable_records(primary_df),
    }


def _train_weighted_model_family(
    model_family: str,
    candidate_params: list[tuple[str, dict[str, Any]]],
    train_X: pd.DataFrame,
    train_y: np.ndarray,
    val_model_X: pd.DataFrame,
    val_model_y: np.ndarray,
    sample_weight: np.ndarray,
) -> dict[str, Any]:
    from sklearn.metrics import roc_auc_score

    best_model = None
    best_name = ""
    best_params: dict[str, Any] = {}
    best_val_auc = float("-inf")
    best_val_model_raw_pd: np.ndarray | None = None
    fit_weight = np.asarray(sample_weight, dtype=float).reshape(-1)

    for candidate_name, params in candidate_params:
        if model_family == "xgboost":
            model = _make_model(params)
            model.fit(
                train_X,
                train_y,
                sample_weight=fit_weight,
                eval_set=[(val_model_X, val_model_y)],
                verbose=False,
            )
            val_model_raw_pd = model.predict_proba(val_model_X)[:, 1]
        elif model_family == "lightgbm":
            model = _make_lgbm_model(params)
            train_matrix = train_X.to_numpy(dtype=float)
            val_model_matrix = val_model_X.to_numpy(dtype=float)
            model.fit(train_matrix, train_y, sample_weight=fit_weight)
            val_model_raw_pd = model.predict_proba(val_model_matrix)[:, 1]
        else:
            raise ValueError(f"Unsupported model family: {model_family}")

        val_auc = float(roc_auc_score(val_model_y, val_model_raw_pd))
        if val_auc > best_val_auc:
            best_model = model
            best_name = candidate_name
            best_params = params
            best_val_auc = val_auc
            best_val_model_raw_pd = val_model_raw_pd

    assert best_model is not None and best_val_model_raw_pd is not None
    return {
        "model": best_model,
        "selected_candidate": best_name,
        "selected_params": best_params,
        "val_model_roc_auc": best_val_auc,
        "val_model_raw_pd": best_val_model_raw_pd,
    }


def _evaluate_fairness_aware_variant(
    spec: FairnessInterventionSpec,
    *,
    bundle: Any,
    features: dict[str, Any],
    train_weights: np.ndarray,
    weighting_details: dict[str, Any],
) -> dict[str, Any]:
    y_train = bundle.train["TARGET"].to_numpy(dtype=int)
    y_val_model = bundle.val_model["TARGET"].to_numpy(dtype=int)
    y_val_policy = bundle.val_policy["TARGET"].to_numpy(dtype=int)
    y_test = bundle.test["TARGET"].to_numpy(dtype=int)

    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)
    scale_pos_weight = float(neg / max(pos, 1))

    xgb_result = _train_weighted_model_family(
        "xgboost",
        _candidate_model_params(scale_pos_weight),
        features["train_full"],
        y_train,
        features["val_model_full"],
        y_val_model,
        train_weights,
    )
    lgbm_result = _train_weighted_model_family(
        "lightgbm",
        _candidate_lgbm_params(scale_pos_weight),
        features["train_full"],
        y_train,
        features["val_model_full"],
        y_val_model,
        train_weights,
    )

    xgb_val_policy_raw = xgb_result["model"].predict_proba(features["val_policy_full"])[:, 1]
    xgb_test_raw = xgb_result["model"].predict_proba(features["test_full"])[:, 1]
    lgbm_val_policy_raw = _predict_lightgbm_raw_pd(lgbm_result["model"], features["val_policy_full"])
    lgbm_test_raw = _predict_lightgbm_raw_pd(lgbm_result["model"], features["test_full"])

    weighted = _select_weighted_average_blend(
        y_val_model,
        xgb_result["val_model_raw_pd"],
        lgbm_result["val_model_raw_pd"],
    )
    weighted_val_policy_raw = (
        weighted["weight_xgb"] * xgb_val_policy_raw
        + weighted["weight_lgbm"] * lgbm_val_policy_raw
    )
    weighted_test_raw = (
        weighted["weight_xgb"] * xgb_test_raw
        + weighted["weight_lgbm"] * lgbm_test_raw
    )
    weighted_eval = _evaluate_calibrated_scores(
        y_val_policy,
        weighted_val_policy_raw,
        y_test,
        weighted_test_raw,
    )
    fairness = _summarize_test_fairness(
        bundle.train,
        bundle.test,
        weighted_eval["calibrator"].predict(weighted_test_raw),
    )

    return {
        "description": spec.description,
        "weighting_strategy": spec.weighting_strategy,
        "weighting_details": weighting_details,
        "component_candidates": {
            "xgboost": {
                "selected_candidate": xgb_result["selected_candidate"],
                "selected_params": xgb_result["selected_params"],
                "val_model_roc_auc": float(xgb_result["val_model_roc_auc"]),
            },
            "lightgbm": {
                "selected_candidate": lgbm_result["selected_candidate"],
                "selected_params": lgbm_result["selected_params"],
                "val_model_roc_auc": float(lgbm_result["val_model_roc_auc"]),
            },
        },
        "weighted_blend": {
            "blend_method": "weighted_average",
            "weights": {
                "xgboost": float(weighted["weight_xgb"]),
                "lightgbm": float(weighted["weight_lgbm"]),
            },
            "val_model_roc_auc": float(weighted["val_model_roc_auc"]),
            "selected_calibrator": weighted_eval["selected_calibrator"],
            "val_policy_calibration_metrics": weighted_eval["val_policy_calibration_metrics"],
            "test_metrics": weighted_eval["metrics"],
        },
        "fairness": fairness,
    }


def _baseline_runtime_summary(bundle: Any, artifact_dir: str, processed_dir: str) -> dict[str, Any]:
    artifacts = load_artifacts(artifact_dir=artifact_dir, processed_dir=processed_dir, strict_artifacts=True)
    report = artifacts.get("reproducibility_report", {})
    model = artifacts["full_model"]
    if not isinstance(model, WeightedBlendModel):
        raise RuntimeError("Expected current FULL runtime candidate to be a WeightedBlendModel.")

    feature_view = (
        report.get("tiers", {})
        .get("FULL", {})
        .get("feature_view", DEFAULT_FULL_FEATURE_VIEW)
    )

    test_X = build_full(bundle.test, artifacts["full_builder"], raw_dir=bundle.raw_dir, feature_view=feature_view)
    test_raw_pd = _predict_raw_pd(model, test_X)
    test_calibrated_pd = np.asarray(artifacts["full_calibrator"].predict(test_raw_pd), dtype=float)

    fairness = _summarize_test_fairness(bundle.train, bundle.test, test_calibrated_pd)
    return {
        "model_version": report.get("full_model_version"),
        "model_family": report.get("model_family"),
        "feature_view": feature_view,
        "val_model_roc_auc": float(report.get("tiers", {}).get("FULL", {}).get("selection", {}).get("val_model_roc_auc", np.nan)),
        "selected_calibrator": report.get("tiers", {}).get("FULL", {}).get("selection", {}).get("calibrator"),
        "val_policy_calibration_metrics": report.get("tiers", {}).get("FULL", {}).get("selection", {}).get("val_policy_calibration_metrics"),
        "test_metrics": _collect_metrics(bundle.test["TARGET"].to_numpy(dtype=int), test_calibrated_pd),
        "fairness": fairness,
        "runtime_candidate": report.get("blend_evaluation", {}).get("best_candidate"),
    }


def run_fairness_aware_modeling_experiments(
    artifact_dir: str = ARTIFACT_DIR,
    processed_dir: str = DATA_DIR,
    raw_dir: str = "data/raw/",
) -> dict[str, Any]:
    if not _has_real_training_inputs(processed_dir, raw_dir):
        raise RuntimeError("Fairness-aware modeling experiments require real processed splits and raw aggregate tables.")

    bundle = _load_real_bundle(processed_dir, raw_dir)
    baseline = _baseline_runtime_summary(bundle, artifact_dir, processed_dir)
    train_groups = _prepare_group_frame(bundle.train, bundle.train)
    full_feature_view = baseline["feature_view"] or DEFAULT_FULL_FEATURE_VIEW

    experiments: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="fairness_aware_build_") as temp_root:
        features = _build_features(
            bundle,
            temp_root,
            processed_dir,
            full_feature_view=full_feature_view,
        )
        y_train = bundle.train["TARGET"].to_numpy(dtype=int)
        for spec in _strategy_specs():
            weights, details = _build_train_weights(spec, train_groups, y_train)
            experiments[spec.name] = _evaluate_fairness_aware_variant(
                spec,
                bundle=bundle,
                features=features,
                train_weights=weights,
                weighting_details=details,
            )

    report = {
        "mode": bundle.mode,
        "offline_only": True,
        "processed_manifest_fingerprint": _processed_manifest_fingerprint(processed_dir),
        "baseline_runtime": baseline,
        "experiments": experiments,
        "notes": [
            "Current runtime artifacts and API behavior remain unchanged.",
            "val_model is used for candidate comparison, val_policy for calibration, and test for final confirmation only.",
            "Interventions are limited to modest region-aware training weights on the existing FULL weighted-blend pipeline.",
        ],
    }
    report_path = _artifact_path(artifact_dir, FAIRNESS_AWARE_MODELING_REPORT_FILENAME)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    report["report_path"] = report_path
    return report


if __name__ == "__main__":
    result = run_fairness_aware_modeling_experiments()
    print(
        json.dumps(
            {
                "report_path": result["report_path"],
                "baseline_runtime": {
                    "model_version": result["baseline_runtime"]["model_version"],
                    "val_model_roc_auc": result["baseline_runtime"]["val_model_roc_auc"],
                    "test_metrics": result["baseline_runtime"]["test_metrics"],
                    "primary_family": {
                        key: result["baseline_runtime"]["fairness"]["families"]["PRIMARY"][key]
                        for key in ["di_pass", "eod_pass", "brier_pass", "di_min", "eod_gap", "brier_ratio"]
                    },
                },
                "experiments": {
                    name: {
                        "val_model_roc_auc": payload["weighted_blend"]["val_model_roc_auc"],
                        "test_metrics": payload["weighted_blend"]["test_metrics"],
                        "primary_family": {
                            key: payload["fairness"]["families"]["PRIMARY"][key]
                            for key in ["di_pass", "eod_pass", "brier_pass", "di_min", "eod_gap", "brier_ratio"]
                        },
                    }
                    for name, payload in result["experiments"].items()
                },
            },
            indent=2,
        )
    )
