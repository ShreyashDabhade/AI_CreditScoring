from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from configs.config import ARTIFACT_DIR, DATA_DIR
from src.fairness_audit import FAIRNESS_COLS, compute_fairness_metrics, derive_fairness_groups
from src.feature_engineering import build_full
from src.models.train import (
    FULL_WEIGHTED_BLEND_MODEL_VERSION,
    _artifact_path,
    _collect_metrics,
    _load_real_bundle,
    _processed_manifest_fingerprint,
    _select_calibrator,
    decision_from_pd,
    load_artifacts,
)
from src.models.runtime_support import WeightedBlendModel

SUBGROUP_CALIBRATION_REPORT_FILENAME = "subgroup_calibration_experiment_report.json"
GROUP_CALIBRATION_MIN_N = 1000
GROUP_CALIBRATION_MIN_DEFAULTS = 50


@dataclass(frozen=True)
class CalibrationStrategySpec:
    name: str
    description: str
    group_column: str | None = None
    allowed_groups: tuple[str, ...] | None = None
    target_selection_policy: str | None = None


def _predict_raw_pd(model: Any, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_raw_pd"):
        raw = model.predict_raw_pd(X)
    else:
        raw = model.predict_proba(X)[:, 1]
    return np.asarray(raw, dtype=float).reshape(-1)


def _prepare_group_frames(train_df: pd.DataFrame, split_df: pd.DataFrame) -> pd.DataFrame:
    fairness_cols_present = [col for col in FAIRNESS_COLS if col in split_df.columns]
    audit_df = split_df[["TARGET"] + fairness_cols_present].copy()
    q1 = float(train_df["AMT_INCOME_TOTAL"].quantile(1 / 3))
    q2 = float(train_df["AMT_INCOME_TOTAL"].quantile(2 / 3))
    return derive_fairness_groups(audit_df, q1, q2)


def _jsonable_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    records = df.replace({np.nan: None}).to_dict(orient="records")
    return [
        {
            key: (value.item() if hasattr(value, "item") else value)
            for key, value in record.items()
        }
        for record in records
    ]


def _fit_grouped_calibration_strategy(
    y_true: np.ndarray,
    raw_pd: np.ndarray,
    groups: pd.Series,
    *,
    min_n: int = GROUP_CALIBRATION_MIN_N,
    min_defaults: int = GROUP_CALIBRATION_MIN_DEFAULTS,
    allowed_groups: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    global_calibrator, global_name, global_fit_metrics = _select_calibrator(y_true, raw_pd)
    groups_series = groups.astype(str).reset_index(drop=True)
    y = np.asarray(y_true, dtype=int).reshape(-1)
    raw = np.asarray(raw_pd, dtype=float).reshape(-1)

    support_df = pd.DataFrame({"group": groups_series, "target": y})
    support_df = (
        support_df.groupby("group")["target"]
        .agg(n="size", defaults="sum")
        .reset_index()
        .sort_values("n", ascending=False)
    )

    group_calibrators: dict[str, Any] = {}
    fitted_groups: list[dict[str, Any]] = []
    fallback_groups: list[dict[str, Any]] = []

    allowed = set(allowed_groups) if allowed_groups is not None else None
    for row in support_df.itertuples(index=False):
        group_value = str(row.group)
        n = int(row.n)
        defaults = int(row.defaults)
        if allowed is not None and group_value not in allowed:
            fallback_groups.append({
                "group": group_value,
                "n": n,
                "defaults": defaults,
                "reason": "not_targeted",
            })
            continue
        if n < min_n or defaults < min_defaults or defaults >= n:
            fallback_groups.append({
                "group": group_value,
                "n": n,
                "defaults": defaults,
                "reason": "support_below_threshold",
            })
            continue
        mask = groups_series == group_value
        calibrator, calibrator_name, fit_metrics = _select_calibrator(y[mask], raw[mask])
        group_calibrators[group_value] = calibrator
        fitted_groups.append({
            "group": group_value,
            "n": n,
            "defaults": defaults,
            "selected_calibrator": calibrator_name,
            "fit_metrics": {
                "roc_auc": float(fit_metrics["roc_auc"]),
                "brier_score": float(fit_metrics["brier_score"]),
            },
        })

    return {
        "global_calibrator": global_calibrator,
        "global_calibrator_name": global_name,
        "global_fit_metrics": {
            "roc_auc": float(global_fit_metrics["roc_auc"]),
            "brier_score": float(global_fit_metrics["brier_score"]),
        },
        "group_calibrators": group_calibrators,
        "fitted_groups": fitted_groups,
        "fallback_groups": fallback_groups,
        "support_thresholds": {
            "min_n": int(min_n),
            "min_defaults": int(min_defaults),
        },
    }


def _apply_grouped_calibration(strategy: dict[str, Any], raw_pd: np.ndarray, groups: pd.Series) -> np.ndarray:
    raw = np.asarray(raw_pd, dtype=float).reshape(-1)
    groups_series = groups.astype(str).reset_index(drop=True)
    calibrated = np.asarray(strategy["global_calibrator"].predict(raw), dtype=float).reshape(-1)
    for group_value, calibrator in strategy["group_calibrators"].items():
        mask = (groups_series == group_value).to_numpy()
        if mask.any():
            calibrated[mask] = np.asarray(calibrator.predict(raw[mask]), dtype=float).reshape(-1)
    return np.clip(calibrated, 0.0, 1.0)


def _summarize_group_calibration(frame: pd.DataFrame, group_col: str, calibrated_pd: np.ndarray) -> list[dict[str, Any]]:
    tmp = frame[[group_col, "TARGET"]].copy()
    tmp["CALIBRATED_PD"] = np.asarray(calibrated_pd, dtype=float)
    rows: list[dict[str, Any]] = []
    for group_value, sub in tmp.groupby(group_col):
        observed = float(sub["TARGET"].mean())
        predicted = float(sub["CALIBRATED_PD"].mean())
        gap = observed - predicted
        rows.append({
            "group": str(group_value),
            "n": int(len(sub)),
            "observed_default_rate": observed,
            "mean_predicted_pd": predicted,
            "signed_gap_obs_minus_pred": float(gap),
            "abs_gap": float(abs(gap)),
            "brier_score": float(np.mean((sub["CALIBRATED_PD"] - sub["TARGET"]) ** 2)),
        })
    rows.sort(key=lambda item: (-item["n"], item["group"]))
    return rows


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
    }
    if len(evaluable) > 0:
        summary.update({
            "di_pass": bool(evaluable["di_pass"].iloc[0]),
            "eod_pass": bool(evaluable["eod_pass"].iloc[0]),
            "brier_pass": bool(evaluable["brier_pass"].iloc[0]),
            "di_min": float(evaluable["di_ratio"].min()),
            "eod_gap": float(evaluable["eod"].max() - evaluable["eod"].min()),
            "brier_ratio": float(evaluable["brier_ratio"].max()),
        })
    summary["metrics_table"] = _jsonable_records(metrics_df)
    return summary


def _summarize_fairness_and_calibration(test_groups: pd.DataFrame, calibrated_pd: np.ndarray) -> dict[str, Any]:
    audit_df = test_groups.copy()
    audit_df["CALIBRATED_PD"] = np.asarray(calibrated_pd, dtype=float)
    audit_df["DECISION"] = [decision_from_pd(float(value)) for value in audit_df["CALIBRATED_PD"]]
    return {
        "families": {
            "PRIMARY": _summarize_fairness_family(audit_df, "FAIR_GROUP_PRIMARY"),
            "SECONDARY": _summarize_fairness_family(audit_df, "FAIR_GROUP_SECONDARY"),
            "TERTIARY": _summarize_fairness_family(audit_df, "FAIR_GROUP_TERTIARY"),
        },
        "subgroup_calibration": {
            "PRIMARY": _summarize_group_calibration(test_groups, "FAIR_GROUP_PRIMARY", calibrated_pd),
            "SECONDARY": _summarize_group_calibration(test_groups, "FAIR_GROUP_SECONDARY", calibrated_pd),
            "TERTIARY": _summarize_group_calibration(test_groups, "FAIR_GROUP_TERTIARY", calibrated_pd),
        },
    }


def _build_improvement_summary(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    baseline_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for family_name, rows in baseline["subgroup_calibration"].items():
        for row in rows:
            baseline_lookup[(family_name, row["group"])] = row

    improvements: list[dict[str, Any]] = []
    for family_name, rows in candidate["subgroup_calibration"].items():
        for row in rows:
            base = baseline_lookup.get((family_name, row["group"]))
            if base is None:
                continue
            improvements.append({
                "family": family_name,
                "group": row["group"],
                "baseline_abs_gap": float(base["abs_gap"]),
                "candidate_abs_gap": float(row["abs_gap"]),
                "abs_gap_improvement": float(base["abs_gap"] - row["abs_gap"]),
                "baseline_brier": float(base["brier_score"]),
                "candidate_brier": float(row["brier_score"]),
                "brier_improvement": float(base["brier_score"] - row["brier_score"]),
            })
    improvements.sort(key=lambda item: (-item["abs_gap_improvement"], -item["brier_improvement"]))
    return {
        "top_abs_gap_improvements": improvements[:8],
    }


def _strategy_specs() -> list[CalibrationStrategySpec]:
    return [
        CalibrationStrategySpec(
            name="global_baseline",
            description="Current persisted global calibrator for the weighted-blend FULL runtime candidate.",
        ),
        CalibrationStrategySpec(
            name="region_aware",
            description="Region-specific calibrators with global fallback for support-stable region groups.",
            group_column="FAIR_GROUP_PRIMARY",
        ),
        CalibrationStrategySpec(
            name="targeted_worst_primary",
            description="Global calibrator plus a targeted override for the worst support-stable primary subgroup.",
            group_column="FAIR_GROUP_PRIMARY",
            target_selection_policy="worst_primary_abs_gap",
        ),
    ]


def _select_target_groups(
    spec: CalibrationStrategySpec,
    *,
    y_true: np.ndarray,
    raw_pd: np.ndarray,
    groups: pd.Series,
) -> tuple[tuple[str, ...] | None, dict[str, Any] | None]:
    if spec.allowed_groups is not None:
        return spec.allowed_groups, {
            "selection_policy": "explicit",
            "target_groups": list(spec.allowed_groups),
        }
    if spec.target_selection_policy != "worst_primary_abs_gap":
        return None, None

    global_calibrator, global_name, global_fit_metrics = _select_calibrator(y_true, raw_pd)
    calibrated = np.asarray(global_calibrator.predict(raw_pd), dtype=float)
    frame = pd.DataFrame(
        {
            "group": groups.astype(str).reset_index(drop=True),
            "TARGET": np.asarray(y_true, dtype=int).reshape(-1),
            "CALIBRATED_PD": calibrated,
        }
    )
    support = (
        frame.groupby("group")["TARGET"]
        .agg(n="size", defaults="sum")
        .reset_index()
    )
    support = support[
        (support["n"] >= GROUP_CALIBRATION_MIN_N)
        & (support["defaults"] >= GROUP_CALIBRATION_MIN_DEFAULTS)
        & (support["defaults"] < support["n"])
    ]
    if support.empty:
        return None, {
            "selection_policy": spec.target_selection_policy,
            "target_groups": [],
            "global_calibrator": global_name,
            "global_fit_metrics": global_fit_metrics,
            "reason": "no_support_stable_primary_group",
        }

    summary = pd.DataFrame(
        _summarize_group_calibration(
            frame.rename(columns={"group": "FAIR_GROUP_PRIMARY"}),
            "FAIR_GROUP_PRIMARY",
            calibrated,
        )
    )
    ranking = support.merge(summary, left_on="group", right_on="group", how="left")
    ranking = ranking.sort_values(
        ["abs_gap", "brier_score", "defaults", "n_x", "group"],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)
    target_group = str(ranking.iloc[0]["group"])
    return (target_group,), {
        "selection_policy": spec.target_selection_policy,
        "target_groups": [target_group],
        "ranking_table": _jsonable_records(ranking),
        "global_calibrator": global_name,
        "global_fit_metrics": {
            "roc_auc": float(global_fit_metrics["roc_auc"]),
            "brier_score": float(global_fit_metrics["brier_score"]),
        },
    }


def _evaluate_strategy(
    spec: CalibrationStrategySpec,
    *,
    baseline_calibrator: Any,
    baseline_calibrator_name: str,
    y_val_policy: np.ndarray,
    val_policy_raw_pd: np.ndarray,
    val_policy_groups: pd.DataFrame,
    y_test: np.ndarray,
    test_raw_pd: np.ndarray,
    test_groups: pd.DataFrame,
) -> dict[str, Any]:
    if spec.group_column is None:
        val_policy_calibrated = np.asarray(baseline_calibrator.predict(val_policy_raw_pd), dtype=float)
        test_calibrated = np.asarray(baseline_calibrator.predict(test_raw_pd), dtype=float)
        details = {
            "selected_calibrator": baseline_calibrator_name,
            "global_fallback_only": True,
            "fitted_groups": [],
            "fallback_groups": [],
            "support_thresholds": {
                "min_n": GROUP_CALIBRATION_MIN_N,
                "min_defaults": GROUP_CALIBRATION_MIN_DEFAULTS,
            },
        }
    else:
        allowed_groups, selection_details = _select_target_groups(
            spec,
            y_true=y_val_policy,
            raw_pd=val_policy_raw_pd,
            groups=val_policy_groups[spec.group_column],
        )
        grouped = _fit_grouped_calibration_strategy(
            y_val_policy,
            val_policy_raw_pd,
            val_policy_groups[spec.group_column],
            min_n=GROUP_CALIBRATION_MIN_N,
            min_defaults=GROUP_CALIBRATION_MIN_DEFAULTS,
            allowed_groups=allowed_groups,
        )
        val_policy_calibrated = _apply_grouped_calibration(grouped, val_policy_raw_pd, val_policy_groups[spec.group_column])
        test_calibrated = _apply_grouped_calibration(grouped, test_raw_pd, test_groups[spec.group_column])
        details = {
            "selected_calibrator": grouped["global_calibrator_name"],
            "global_calibrator_fit_metrics": grouped["global_fit_metrics"],
            "global_fallback_only": False,
            "group_column": spec.group_column,
            "targeted_groups": list(allowed_groups) if allowed_groups is not None else None,
            "fitted_groups": grouped["fitted_groups"],
            "fallback_groups": grouped["fallback_groups"],
            "support_thresholds": grouped["support_thresholds"],
        }
        if selection_details is not None:
            details["target_selection"] = selection_details

    result = {
        "description": spec.description,
        "val_policy_metrics": _collect_metrics(y_val_policy, val_policy_calibrated),
        "test_metrics": _collect_metrics(y_test, test_calibrated),
    }
    result.update(details)
    result.update(_summarize_fairness_and_calibration(test_groups, test_calibrated))
    return result


def _verify_weighted_blend_runtime(artifacts: dict[str, Any]) -> None:
    model = artifacts["full_model"]
    report = artifacts.get("reproducibility_report", {})
    if not isinstance(model, WeightedBlendModel):
        raise RuntimeError("Subgroup calibration prototype requires the current FULL runtime model to be a WeightedBlendModel.")
    if report.get("full_model_version") != FULL_WEIGHTED_BLEND_MODEL_VERSION:
        raise RuntimeError(
            f"Expected FULL runtime version {FULL_WEIGHTED_BLEND_MODEL_VERSION}, got {report.get('full_model_version')!r}."
        )


def run_subgroup_calibration_experiments(
    artifact_dir: str = ARTIFACT_DIR,
    processed_dir: str = DATA_DIR,
    raw_dir: str = "data/raw/",
) -> dict[str, Any]:
    bundle = _load_real_bundle(processed_dir, raw_dir)
    artifacts = load_artifacts(artifact_dir=artifact_dir, processed_dir=processed_dir, strict_artifacts=True)
    _verify_weighted_blend_runtime(artifacts)

    val_policy_X = build_full(bundle.val_policy, artifacts["full_builder"], raw_dir=raw_dir)
    test_X = build_full(bundle.test, artifacts["full_builder"], raw_dir=raw_dir)

    y_val_policy = bundle.val_policy["TARGET"].to_numpy(dtype=int)
    y_test = bundle.test["TARGET"].to_numpy(dtype=int)
    val_policy_raw_pd = _predict_raw_pd(artifacts["full_model"], val_policy_X)
    test_raw_pd = _predict_raw_pd(artifacts["full_model"], test_X)

    val_policy_groups = _prepare_group_frames(bundle.train, bundle.val_policy)
    test_groups = _prepare_group_frames(bundle.train, bundle.test)

    report = artifacts.get("reproducibility_report", {})
    baseline_name = (
        report.get("tiers", {})
        .get("FULL", {})
        .get("selection", {})
        .get("calibrator", type(artifacts["full_calibrator"]).__name__)
    )

    strategy_results: dict[str, Any] = {}
    for spec in _strategy_specs():
        strategy_results[spec.name] = _evaluate_strategy(
            spec,
            baseline_calibrator=artifacts["full_calibrator"],
            baseline_calibrator_name=str(baseline_name),
            y_val_policy=y_val_policy,
            val_policy_raw_pd=val_policy_raw_pd,
            val_policy_groups=val_policy_groups,
            y_test=y_test,
            test_raw_pd=test_raw_pd,
            test_groups=test_groups,
        )

    baseline = strategy_results["global_baseline"]
    for name, result in strategy_results.items():
        if name == "global_baseline":
            continue
        result["improvement_vs_global_baseline"] = _build_improvement_summary(baseline, result)

    summary = {
        "mode": bundle.mode,
        "processed_manifest_fingerprint": _processed_manifest_fingerprint(processed_dir),
        "full_model_version": report.get("full_model_version"),
        "model_family": report.get("model_family"),
        "runtime_candidate": report.get("blend_evaluation", {}).get("best_candidate"),
        "offline_only": True,
        "support_thresholds": {
            "min_n": GROUP_CALIBRATION_MIN_N,
            "min_defaults": GROUP_CALIBRATION_MIN_DEFAULTS,
        },
        "strategies": strategy_results,
        "notes": [
            "Weighted-blend raw scores remain fixed across all strategies.",
            "val_policy is used only for calibration fitting; test is final confirmation only.",
            "This experiment does not modify the deployed runtime stack or fairness artifact ownership.",
        ],
    }
    report_path = _artifact_path(artifact_dir, SUBGROUP_CALIBRATION_REPORT_FILENAME)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    summary["report_path"] = report_path
    return summary


if __name__ == "__main__":
    result = run_subgroup_calibration_experiments()
    print(json.dumps({
        "report_path": result["report_path"],
        "full_model_version": result["full_model_version"],
        "strategies": {
            name: {
                "test_metrics": payload["test_metrics"],
                "fairness": {
                    family: {
                        key: payload["families"][family][key]
                        for key in ["di_pass", "eod_pass", "brier_pass", "di_min", "eod_gap", "brier_ratio"]
                    }
                    for family in ["PRIMARY", "SECONDARY", "TERTIARY"]
                },
            }
            for name, payload in result["strategies"].items()
        },
    }, indent=2))
