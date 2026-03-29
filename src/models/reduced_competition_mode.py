from __future__ import annotations

import json
import os
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from configs.config import RANDOM_STATE
from src.data_pipeline import apply_income_cap, prepare_application_frame
from src.feature_engineering import build_reduced, fit_reduced_builder
from src.models.train import (
    _artifact_path,
    _has_real_training_inputs,
    _load_real_bundle,
    _processed_manifest_fingerprint,
    _train_tier_model,
    load_artifacts,
)
from src.runtime_verification import load_json_object

COMPETITION_REPORT_FILENAME = "reduced_competition_report.json"
DEFAULT_COMPETITION_PROCESSED_DIR = "data/processed_random_stratified"
DEFAULT_COMPETITION_ARTIFACT_DIR = "artifacts/random_stratified_reduced"
DEFAULT_PREDICTION_FILENAME = "application_test_predictions.csv"


def _scored_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "brier_score": float(brier_score_loss(y_true, scores)),
    }


def _split_summary(bundle: Any) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for split_name in ("train", "val_model", "val_policy", "test"):
        split_df = getattr(bundle, split_name)
        summary[split_name] = {
            "rows": int(len(split_df)),
            "target_rate": float(split_df["TARGET"].mean()),
            "positive_count": int(split_df["TARGET"].sum()),
        }
    return summary


def _load_processed_manifest(processed_dir: str) -> dict[str, Any]:
    return load_json_object(
        os.path.join(processed_dir, "processed_artifact_manifest.json"),
        "processed manifest",
    )


def _baseline_proxy_time_comparison(raw_dir: str) -> dict[str, Any] | None:
    baseline_processed_dir = "data/processed"
    baseline_artifact_dir = "artifacts"
    baseline_report_path = os.path.join(baseline_artifact_dir, "reproducibility_report.json")
    if not (
        _has_real_training_inputs(baseline_processed_dir, raw_dir)
        and os.path.exists(baseline_report_path)
    ):
        return None

    baseline_bundle = _load_real_bundle(baseline_processed_dir, raw_dir)
    artifacts = load_artifacts(
        artifact_dir=baseline_artifact_dir,
        processed_dir=baseline_processed_dir,
        strict_artifacts=True,
    )
    baseline_report = load_json_object(
        baseline_report_path,
        "proxy-time reproducibility report",
    )

    test_X = build_reduced(baseline_bundle.test, artifacts["reduced_builder"])
    test_raw = np.asarray(artifacts["reduced_model"].predict_proba(test_X), dtype=float)[:, 1]
    test_calibrated = np.asarray(artifacts["reduced_calibrator"].predict(test_raw), dtype=float)
    test_y = baseline_bundle.test["TARGET"].to_numpy(dtype=int)

    return {
        "split_mode": "proxy_time",
        "processed_dir": baseline_processed_dir,
        "artifact_dir": baseline_artifact_dir,
        "val_model_roc_auc": float(
            baseline_report["tiers"]["REDUCED"]["selection"]["val_model_roc_auc"]
        ),
        "test_metrics": _scored_metrics(test_y, test_calibrated),
    }


def run_reduced_competition_mode(
    *,
    processed_dir: str = DEFAULT_COMPETITION_PROCESSED_DIR,
    artifact_dir: str = DEFAULT_COMPETITION_ARTIFACT_DIR,
    raw_dir: str = "data/raw",
    prediction_filename: str = DEFAULT_PREDICTION_FILENAME,
) -> dict[str, Any]:
    if not _has_real_training_inputs(processed_dir, raw_dir):
        raise RuntimeError(
            f"Reduced competition mode requires real processed splits and raw aggregate tables: {processed_dir}"
        )

    os.makedirs(artifact_dir, exist_ok=True)
    manifest = _load_processed_manifest(processed_dir)
    bundle = _load_real_bundle(processed_dir, raw_dir)
    reduced_builder = fit_reduced_builder(
        bundle.train,
        save_path=_artifact_path(artifact_dir, "reduced_feature_builder.joblib"),
        processed_manifest_path=os.path.join(processed_dir, "processed_artifact_manifest.json"),
    )
    train_reduced = build_reduced(bundle.train, reduced_builder)
    val_model_reduced = build_reduced(bundle.val_model, reduced_builder)
    val_policy_reduced = build_reduced(bundle.val_policy, reduced_builder)
    test_reduced = build_reduced(bundle.test, reduced_builder)

    y_train = bundle.train["TARGET"].to_numpy(dtype=int)
    y_val_model = bundle.val_model["TARGET"].to_numpy(dtype=int)
    y_val_policy = bundle.val_policy["TARGET"].to_numpy(dtype=int)
    y_test = bundle.test["TARGET"].to_numpy(dtype=int)

    reduced_result = _train_tier_model(
        "REDUCED",
        train_reduced,
        y_train,
        val_model_reduced,
        y_val_model,
        val_policy_reduced,
        y_val_policy,
        test_reduced,
        y_test,
        artifact_dir,
    )

    val_model_raw = np.asarray(
        reduced_result["model"].predict_proba(val_model_reduced),
        dtype=float,
    )[:, 1]
    test_raw = np.asarray(
        reduced_result["model"].predict_proba(test_reduced),
        dtype=float,
    )[:, 1]
    test_calibrated = np.asarray(
        reduced_result["calibrator"].predict(test_raw),
        dtype=float,
    )

    income_cap = float(joblib.load(os.path.join(processed_dir, "income_cap.joblib")))
    application_test = pd.read_csv(os.path.join(raw_dir, "application_test.csv"))
    application_test_has_target = "TARGET" in application_test.columns
    application_test = prepare_application_frame(application_test)
    application_test = apply_income_cap(application_test, income_cap)
    application_test_reduced = build_reduced(application_test, reduced_builder)

    application_test_raw = np.asarray(
        reduced_result["model"].predict_proba(application_test_reduced),
        dtype=float,
    )[:, 1]
    application_test_calibrated = np.asarray(
        reduced_result["calibrator"].predict(application_test_raw),
        dtype=float,
    )
    prediction_path = _artifact_path(artifact_dir, prediction_filename)
    pd.DataFrame(
        {
            "SK_ID_CURR": application_test["SK_ID_CURR"].to_numpy(dtype=np.int64, copy=False),
            "TARGET_PROBABILITY": application_test_calibrated,
        }
    ).to_csv(prediction_path, index=False)

    report = {
        "split_mode": manifest.get("split_mode"),
        "random_state": int(manifest.get("random_state", RANDOM_STATE)),
        "processed_dir": processed_dir,
        "artifact_dir": artifact_dir,
        "processed_manifest_fingerprint": _processed_manifest_fingerprint(processed_dir),
        "split_summary": _split_summary(bundle),
        "reduced_model": {
            "selected_candidate": reduced_result["selected_candidate"],
            "selected_params": reduced_result["selected_params"],
            "selected_calibrator": reduced_result["selected_calibrator"],
            "val_model_metrics": {
                "roc_auc": float(roc_auc_score(y_val_model, val_model_raw)),
            },
            "val_policy_calibration_metrics": reduced_result["val_policy_calibration_metrics"],
            "test_metrics": _scored_metrics(y_test, test_calibrated),
        },
        "application_test": {
            "rows": int(len(application_test)),
            "has_target": application_test_has_target,
            "auc_computed": False,
            "prediction_path": prediction_path,
        },
        "proxy_time_comparison": _baseline_proxy_time_comparison(raw_dir),
    }
    report_path = _artifact_path(artifact_dir, COMPETITION_REPORT_FILENAME)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    report["report_path"] = report_path
    return report


if __name__ == "__main__":
    result = run_reduced_competition_mode(
        processed_dir=os.environ.get("DATA_PROCESSED_DIR", DEFAULT_COMPETITION_PROCESSED_DIR),
        artifact_dir=os.environ.get("ARTIFACT_DIR", DEFAULT_COMPETITION_ARTIFACT_DIR),
        raw_dir=os.environ.get("DATA_RAW_DIR", "data/raw"),
        prediction_filename=os.environ.get("PREDICTION_FILENAME", DEFAULT_PREDICTION_FILENAME),
    )
    print(json.dumps(result, indent=2))
