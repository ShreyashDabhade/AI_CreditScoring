from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


EXPERIMENT_REPORT_FILENAME = "subgroup_calibration_experiment_report.json"
REPRO_REPORT_FILENAME = "reproducibility_report.json"
LIVE_FAIRNESS_ARTIFACT_FILENAMES = (
    "fairness_optimized_model.joblib",
    "fairness_optimized_calibrator.joblib",
    "fairness_optimized_shap_explainer.joblib",
)
FAMILY_PATHS = {
    "PRIMARY": "audit_primary.csv",
    "SECONDARY": "audit_secondary.csv",
    "TERTIARY": "audit_tertiary.csv",
}
FAMILY_LABELS = {
    "PRIMARY": "Primary proxy family",
    "SECONDARY": "Secondary proxy family",
    "TERTIARY": "Tertiary proxy family",
}


def _safe_float(value: Any) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _safe_bool(value: Any) -> bool | None:
    text = str(value or "").strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else None


def _format_metric(value: float | None, *, pct: bool = False) -> str:
    if value is None:
        return "—"
    if pct:
        return f"{value * 100:.1f}%"
    return f"{value:.3f}"


def _family_summary_from_csv(path: Path, family_name: str) -> dict[str, Any]:
    if not path.exists():
        return {
            "family": family_name,
            "label": FAMILY_LABELS[family_name],
            "available": False,
            "source": str(path),
            "message": "Audit table not found.",
        }

    with path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    evaluable_rows = [row for row in rows if _safe_bool(row.get("evaluable")) is True]
    di_values = [_safe_float(row.get("di_ratio")) for row in evaluable_rows]
    di_values = [value for value in di_values if value is not None]
    eod_values = [_safe_float(row.get("eod")) for row in evaluable_rows]
    eod_values = [value for value in eod_values if value is not None]
    brier_values = [_safe_float(row.get("brier_ratio")) for row in evaluable_rows]
    brier_values = [value for value in brier_values if value is not None]

    di_min = min(di_values) if di_values else None
    eod_gap = (max(eod_values) - min(eod_values)) if eod_values else None
    brier_ratio = max(brier_values) if brier_values else None
    di_pass = all(_safe_bool(row.get("di_pass")) is True for row in evaluable_rows) if evaluable_rows else None
    eod_pass = all(_safe_bool(row.get("eod_pass")) is True for row in evaluable_rows) if evaluable_rows else None
    brier_pass = all(_safe_bool(row.get("brier_pass")) is True for row in evaluable_rows) if evaluable_rows else None
    family_pass = bool(di_pass and eod_pass and brier_pass) if evaluable_rows else None

    return {
        "family": family_name,
        "label": FAMILY_LABELS[family_name],
        "available": True,
        "source": str(path),
        "rows": len(rows),
        "evaluable_cells": len(evaluable_rows),
        "di_min": di_min,
        "di_min_text": _format_metric(di_min),
        "eod_gap": eod_gap,
        "eod_gap_text": _format_metric(eod_gap, pct=True),
        "brier_ratio": brier_ratio,
        "brier_ratio_text": _format_metric(brier_ratio),
        "di_pass": di_pass,
        "eod_pass": eod_pass,
        "brier_pass": brier_pass,
        "family_pass": family_pass,
    }


def _build_standard_summary(fairness_dir: str | Path) -> dict[str, Any]:
    directory = Path(fairness_dir)
    families = [
        _family_summary_from_csv(directory / filename, family_name)
        for family_name, filename in FAMILY_PATHS.items()
    ]
    evaluable = [item for item in families if item.get("evaluable_cells", 0) > 0]
    passed = [item for item in evaluable if item.get("family_pass") is True]
    status = "unavailable"
    if evaluable:
        status = "stable" if len(passed) == len(evaluable) else "watch"

    return {
        "label": "Standard Model",
        "subtitle": "Current deployed proxy fairness audit",
        "status": status,
        "families_evaluable": len(evaluable),
        "families_passed": len(passed),
        "headline": (
            f"{len(passed)} of {len(evaluable)} evaluable proxy families passed all fairness checks."
            if evaluable
            else "No evaluable fairness audit families are available."
        ),
        "families": families,
    }


def _summarize_family_payload(family_name: str, payload: dict[str, Any]) -> dict[str, Any]:
    di_min = _safe_float(payload.get("di_min"))
    eod_gap = _safe_float(payload.get("eod_gap"))
    brier_ratio = _safe_float(payload.get("brier_ratio"))
    di_pass = payload.get("di_pass") if isinstance(payload.get("di_pass"), bool) else _safe_bool(payload.get("di_pass"))
    eod_pass = payload.get("eod_pass") if isinstance(payload.get("eod_pass"), bool) else _safe_bool(payload.get("eod_pass"))
    brier_pass = payload.get("brier_pass") if isinstance(payload.get("brier_pass"), bool) else _safe_bool(payload.get("brier_pass"))

    return {
        "family": family_name,
        "label": FAMILY_LABELS.get(family_name, family_name.title()),
        "available": True,
        "rows": int(payload.get("rows", 0) or 0),
        "evaluable_cells": int(payload.get("evaluable_cells", 0) or 0),
        "di_min": di_min,
        "di_min_text": _format_metric(di_min),
        "eod_gap": eod_gap,
        "eod_gap_text": _format_metric(eod_gap, pct=True),
        "brier_ratio": brier_ratio,
        "brier_ratio_text": _format_metric(brier_ratio),
        "di_pass": di_pass,
        "eod_pass": eod_pass,
        "brier_pass": brier_pass,
        "family_pass": bool(di_pass and eod_pass and brier_pass) if payload.get("evaluable_cells") else None,
    }


def _strategy_rank(strategy: dict[str, Any]) -> tuple[int, float, float, float]:
    families = strategy.get("families", {}) if isinstance(strategy.get("families"), dict) else {}
    summaries = [_summarize_family_payload(name, payload) for name, payload in families.items() if isinstance(payload, dict)]
    evaluable = [item for item in summaries if item.get("evaluable_cells", 0) > 0]
    pass_count = sum(1 for item in evaluable if item.get("family_pass") is True)
    di_floor = min((item["di_min"] for item in evaluable if item.get("di_min") is not None), default=0.0)
    eod_gap = min((item["eod_gap"] for item in evaluable if item.get("eod_gap") is not None), default=999.0)
    brier = min((item["brier_ratio"] for item in evaluable if item.get("brier_ratio") is not None), default=999.0)
    return (pass_count, di_floor, -eod_gap, -brier)


def _build_tradeoff_notes(
    baseline: dict[str, Any] | None,
    candidate: dict[str, Any],
) -> list[str]:
    notes: list[str] = []
    baseline_metrics = baseline.get("test_metrics", {}) if isinstance(baseline, dict) else {}
    candidate_metrics = candidate.get("test_metrics", {}) if isinstance(candidate.get("test_metrics"), dict) else {}

    base_auc = _safe_float(baseline_metrics.get("roc_auc"))
    cand_auc = _safe_float(candidate_metrics.get("roc_auc"))
    if base_auc is not None and cand_auc is not None:
        delta_auc = cand_auc - base_auc
        notes.append(
            f"Offline ROC-AUC changed by {delta_auc:+.4f} versus the experiment baseline."
        )

    base_brier = _safe_float(baseline_metrics.get("brier_score"))
    cand_brier = _safe_float(candidate_metrics.get("brier_score"))
    if base_brier is not None and cand_brier is not None:
        delta_brier = cand_brier - base_brier
        notes.append(
            f"Offline Brier score changed by {delta_brier:+.4f}; lower values indicate better calibration."
        )

    description = str(candidate.get("description", "")).strip()
    if description:
        notes.append(description)

    return notes


def _build_experimental_summary(artifact_dir: str | Path) -> dict[str, Any]:
    artifact_root = Path(artifact_dir)
    report_path = artifact_root / EXPERIMENT_REPORT_FILENAME
    report = _load_json(report_path)
    if not report:
        return {
            "available": False,
            "label": "Fairness-Optimized Model",
            "subtitle": "Offline experimental comparison",
            "source": str(report_path),
            "message": "No subgroup calibration experiment report is present in the current artifacts, so only the standard deployed audit can be shown.",
            "artifact_files": [],
        }

    strategies = report.get("strategies", {}) if isinstance(report.get("strategies"), dict) else {}
    baseline = strategies.get("global_baseline") if isinstance(strategies.get("global_baseline"), dict) else None
    candidates = [
        (name, payload)
        for name, payload in strategies.items()
        if name != "global_baseline" and isinstance(payload, dict)
    ]
    if not candidates:
        return {
            "available": False,
            "label": "Fairness-Optimized Model",
            "subtitle": "Offline experimental comparison",
            "source": str(report_path),
            "message": "Experiment report is present but does not contain a comparison candidate beyond the baseline strategy.",
            "artifact_files": [str(report_path)],
        }

    selected_name, selected_payload = max(candidates, key=lambda item: _strategy_rank(item[1]))
    families = selected_payload.get("families", {}) if isinstance(selected_payload.get("families"), dict) else {}
    family_summaries = [
        _summarize_family_payload(family_name, payload)
        for family_name, payload in families.items()
        if isinstance(payload, dict)
    ]
    evaluable = [item for item in family_summaries if item.get("evaluable_cells", 0) > 0]
    passed = [item for item in evaluable if item.get("family_pass") is True]

    return {
        "available": True,
        "label": "Fairness-Optimized Model",
        "subtitle": "Offline experimental comparison",
        "source": str(report_path),
        "strategy_name": selected_name,
        "headline": (
            f"{len(passed)} of {len(evaluable)} evaluable proxy families passed all fairness checks in the selected experiment."
            if evaluable
            else "The selected experiment does not contain evaluable proxy-family summaries."
        ),
        "offline_only": bool(report.get("offline_only", True)),
        "runtime_candidate": report.get("runtime_candidate"),
        "families": family_summaries,
        "tradeoff_notes": _build_tradeoff_notes(baseline, selected_payload),
        "artifact_files": [str(report_path)],
    }


def _detect_live_fairness_toggle(artifact_dir: str | Path) -> dict[str, Any]:
    artifact_root = Path(artifact_dir)
    repro_path = artifact_root / REPRO_REPORT_FILENAME
    repro = _load_json(repro_path) or {}
    required_files = [artifact_root / name for name in LIVE_FAIRNESS_ARTIFACT_FILENAMES]
    live_version = str(repro.get("fairness_optimized_model_version", "")).strip()
    live_flag = bool(repro.get("fairness_optimized_deployed"))
    has_files = all(path.exists() for path in required_files)
    available = bool(live_version and live_flag and has_files)

    return {
        "available": available,
        "model_version": live_version or None,
        "artifact_files": [str(path) for path in required_files if path.exists()],
        "message": (
            "A second live fairness-optimized scoring path is explicitly declared in the runtime artifacts."
            if available
            else "No explicit second deployed fairness-optimized runtime path was found in the current artifacts."
        ),
    }


def build_fairness_comparison_snapshot(
    artifact_dir: str | Path,
    fairness_dir: str | Path,
) -> dict[str, Any]:
    live_toggle = _detect_live_fairness_toggle(artifact_dir)
    standard = _build_standard_summary(fairness_dir)
    experimental = _build_experimental_summary(artifact_dir)
    mode = "live_toggle" if live_toggle["available"] else "offline_comparison"

    artifact_files = []
    artifact_files.extend(
        family.get("source")
        for family in standard.get("families", [])
        if family.get("source")
    )
    artifact_files.extend(live_toggle.get("artifact_files", []))
    artifact_files.extend(experimental.get("artifact_files", []))

    return {
        "mode": mode,
        "mode_label": "True live toggle" if mode == "live_toggle" else "Offline comparison fallback",
        "live_toggle": live_toggle,
        "standard": standard,
        "experimental": experimental,
        "tradeoff_explanation": (
            "The current deployed model stays unchanged. The comparison panel uses offline fairness artifacts unless a second live runtime is explicitly present."
            if mode == "offline_comparison"
            else "The comparison can switch between the deployed standard path and a separately deployed fairness-optimized runtime."
        ),
        "artifact_files": sorted(dict.fromkeys(artifact_files)),
    }
