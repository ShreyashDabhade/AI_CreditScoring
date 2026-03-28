from __future__ import annotations

import json
import tempfile
from typing import Any

import numpy as np

from configs.config import ARTIFACT_DIR, DATA_DIR
from src.feature_engineering import build_reduced
from src.models.train import (
    _artifact_path,
    _build_features,
    _build_training_regime_bundle,
    _collect_metrics,
    _has_real_training_inputs,
    _load_real_bundle,
    _processed_manifest_fingerprint,
    _train_tier_model,
    load_artifacts,
)

REDUCED_TRAINING_REGIME_COMPARE_REPORT_FILENAME = "reduced_training_regime_comparison_report.json"


def _compare_metric_deltas(
    baseline_metrics: dict[str, float],
    candidate_metrics: dict[str, float],
) -> dict[str, float]:
    return {
        "test_auc_delta": float(candidate_metrics["roc_auc"] - baseline_metrics["roc_auc"]),
        "test_brier_delta": float(candidate_metrics["brier_score"] - baseline_metrics["brier_score"]),
        "test_accuracy_delta": float(candidate_metrics["accuracy"] - baseline_metrics["accuracy"]),
    }


def _runtime_reduced_summary(
    bundle: Any,
    artifact_dir: str,
    processed_dir: str,
) -> dict[str, Any]:
    artifacts = load_artifacts(artifact_dir=artifact_dir, processed_dir=processed_dir, strict_artifacts=True)
    report = artifacts.get("reproducibility_report", {})
    reduced_tier = report.get("tiers", {}).get("REDUCED", {})
    test_X = build_reduced(bundle.test, artifacts["reduced_builder"])
    test_raw_pd = artifacts["reduced_model"].predict_proba(test_X)[:, 1]
    test_calibrated_pd = np.asarray(artifacts["reduced_calibrator"].predict(test_raw_pd), dtype=float)
    return {
        "model_version": reduced_tier.get("model_version", report.get("reduced_model_version")),
        "feature_count": reduced_tier.get("feature_count"),
        "selected_candidate": reduced_tier.get("selection", {}).get("candidate"),
        "selected_calibrator": reduced_tier.get("selection", {}).get("calibrator"),
        "val_model_roc_auc": reduced_tier.get("selection", {}).get("val_model_roc_auc"),
        "val_policy_calibration_metrics": reduced_tier.get("selection", {}).get("val_policy_calibration_metrics"),
        "test_metrics": _collect_metrics(bundle.test["TARGET"].to_numpy(dtype=int), test_calibrated_pd),
    }


def _train_reduced_regime(
    bundle: Any,
    *,
    processed_dir: str,
    training_regime: str,
) -> dict[str, Any]:
    regime_bundle, regime_details = _build_training_regime_bundle(bundle, training_regime)

    with tempfile.TemporaryDirectory(prefix=f"reduced_regime_{training_regime}_") as temp_root:
        features = _build_features(regime_bundle, temp_root, processed_dir)
        y_train = regime_bundle.train["TARGET"].to_numpy(dtype=int)
        y_val_model = regime_bundle.val_model["TARGET"].to_numpy(dtype=int)
        y_val_policy = regime_bundle.val_policy["TARGET"].to_numpy(dtype=int)
        y_test = regime_bundle.test["TARGET"].to_numpy(dtype=int)
        result = _train_tier_model(
            "REDUCED",
            features["train_reduced"],
            y_train,
            features["val_model_reduced"],
            y_val_model,
            features["val_policy_reduced"],
            y_val_policy,
            features["test_reduced"],
            y_test,
            temp_root,
        )

    return {
        "regime_details": regime_details,
        "sample_counts": {
            "train": int(len(regime_bundle.train)),
            "val_model": int(len(regime_bundle.val_model)),
            "calibration_holdout": int(len(regime_bundle.val_policy)),
            "test": int(len(regime_bundle.test)),
        },
        "selected_candidate": result["selected_candidate"],
        "selected_params": result["selected_params"],
        "selected_calibrator": result["selected_calibrator"],
        "val_model_roc_auc": float(result["val_model_roc_auc"]),
        "val_policy_calibration_metrics": result["val_policy_calibration_metrics"],
        "test_metrics": result["metrics"],
    }


def run_reduced_training_regime_comparison(
    artifact_dir: str = ARTIFACT_DIR,
    processed_dir: str = DATA_DIR,
    raw_dir: str = "data/raw/",
) -> dict[str, Any]:
    if not _has_real_training_inputs(processed_dir, raw_dir):
        raise RuntimeError("Reduced training-regime comparison requires real processed splits and raw aggregate tables.")

    bundle = _load_real_bundle(processed_dir, raw_dir)
    runtime_baseline = _runtime_reduced_summary(bundle, artifact_dir, processed_dir)
    train_only = _train_reduced_regime(bundle, processed_dir=processed_dir, training_regime="train_only")
    train_plus_val_policy = _train_reduced_regime(
        bundle,
        processed_dir=processed_dir,
        training_regime="train_plus_val_policy",
    )

    report = {
        "mode": bundle.mode,
        "offline_only": True,
        "processed_manifest_fingerprint": _processed_manifest_fingerprint(processed_dir),
        "runtime_baseline": runtime_baseline,
        "training_regimes": {
            "train_only": train_only,
            "train_plus_val_policy": train_plus_val_policy,
        },
        "delta_vs_runtime": {
            "train_only": _compare_metric_deltas(runtime_baseline["test_metrics"], train_only["test_metrics"]),
            "train_plus_val_policy": _compare_metric_deltas(runtime_baseline["test_metrics"], train_plus_val_policy["test_metrics"]),
        },
        "delta_merged_vs_train_only": _compare_metric_deltas(
            train_only["test_metrics"],
            train_plus_val_policy["test_metrics"],
        ),
        "notes": [
            "This is an offline-only REDUCED comparison and does not overwrite reduced_model.joblib or reduced_calibrator.joblib.",
            "train_only keeps the canonical val_model -> val_policy -> test order.",
            "train_plus_val_policy merges val_policy into fitting and reuses val_model as the last non-test selection/calibration holdout.",
        ],
    }
    report_path = _artifact_path(artifact_dir, REDUCED_TRAINING_REGIME_COMPARE_REPORT_FILENAME)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    report["report_path"] = report_path
    return report


if __name__ == "__main__":
    result = run_reduced_training_regime_comparison()
    print(
        json.dumps(
            {
                "report_path": result["report_path"],
                "runtime_baseline": result["runtime_baseline"]["test_metrics"],
                "train_only": result["training_regimes"]["train_only"]["test_metrics"],
                "train_plus_val_policy": result["training_regimes"]["train_plus_val_policy"]["test_metrics"],
                "delta_merged_vs_train_only": result["delta_merged_vs_train_only"],
            },
            indent=2,
        )
    )
