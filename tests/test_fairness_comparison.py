from __future__ import annotations

import json
from pathlib import Path

from src.fairness_comparison import build_fairness_comparison_snapshot


def _write_audit_csv(path: Path, rows: list[dict[str, object]]) -> None:
    header = [
        "group_value",
        "n",
        "n_defaults",
        "approve_rate",
        "tpr",
        "brier_score",
        "di_ratio",
        "eod",
        "brier_ratio",
        "evaluable",
        "di_pass",
        "eod_pass",
        "brier_pass",
    ]
    lines = [",".join(header)]
    for row in rows:
        lines.append(",".join(str(row.get(column, "")) for column in header))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _seed_standard_audit(fairness_dir: Path) -> None:
    fairness_dir.mkdir(parents=True, exist_ok=True)
    _write_audit_csv(
        fairness_dir / "audit_primary.csv",
        [
            {
                "group_value": "REGION_1",
                "n": 300,
                "n_defaults": 120,
                "approve_rate": 0.16,
                "tpr": 0.14,
                "brier_score": 0.31,
                "di_ratio": 0.91,
                "eod": -0.03,
                "brier_ratio": 1.01,
                "evaluable": True,
                "di_pass": True,
                "eod_pass": True,
                "brier_pass": True,
            },
            {
                "group_value": "REGION_2",
                "n": 305,
                "n_defaults": 130,
                "approve_rate": 0.17,
                "tpr": 0.15,
                "brier_score": 0.30,
                "di_ratio": 0.95,
                "eod": 0.01,
                "brier_ratio": 1.03,
                "evaluable": True,
                "di_pass": True,
                "eod_pass": True,
                "brier_pass": True,
            },
        ],
    )
    _write_audit_csv(
        fairness_dir / "audit_secondary.csv",
        [
            {
                "group_value": "INCOME_T1",
                "n": 200,
                "n_defaults": 90,
                "approve_rate": 0.13,
                "tpr": 0.12,
                "brier_score": 0.34,
                "di_ratio": 0.74,
                "eod": -0.05,
                "brier_ratio": 1.11,
                "evaluable": True,
                "di_pass": False,
                "eod_pass": True,
                "brier_pass": True,
            },
            {
                "group_value": "INCOME_T2",
                "n": 210,
                "n_defaults": 96,
                "approve_rate": 0.17,
                "tpr": 0.15,
                "brier_score": 0.31,
                "di_ratio": 1.0,
                "eod": 0.0,
                "brier_ratio": 1.0,
                "evaluable": True,
                "di_pass": False,
                "eod_pass": True,
                "brier_pass": True,
            },
        ],
    )
    _write_audit_csv(
        fairness_dir / "audit_tertiary.csv",
        [
            {
                "group_value": "N_Y__2_PLUS",
                "n": 80,
                "n_defaults": 35,
                "approve_rate": "",
                "tpr": "",
                "brier_score": "",
                "di_ratio": "",
                "eod": "",
                "brier_ratio": "",
                "evaluable": False,
                "di_pass": "",
                "eod_pass": "",
                "brier_pass": "",
            }
        ],
    )


def test_build_fairness_comparison_snapshot_uses_offline_fallback_when_no_live_runtime(tmp_path: Path):
    artifact_dir = tmp_path / "artifacts"
    fairness_dir = tmp_path / "fairness"
    artifact_dir.mkdir()
    _seed_standard_audit(fairness_dir)

    snapshot = build_fairness_comparison_snapshot(artifact_dir, fairness_dir)

    assert snapshot["mode"] == "offline_comparison"
    assert snapshot["live_toggle"]["available"] is False
    assert snapshot["standard"]["families_evaluable"] == 2
    assert snapshot["experimental"]["available"] is False
    assert "No subgroup calibration experiment report" in snapshot["experimental"]["message"]


def test_build_fairness_comparison_snapshot_reads_offline_experiment_report(tmp_path: Path):
    artifact_dir = tmp_path / "artifacts"
    fairness_dir = tmp_path / "fairness"
    artifact_dir.mkdir()
    _seed_standard_audit(fairness_dir)

    report = {
        "offline_only": True,
        "runtime_candidate": "weighted_blend_full",
        "strategies": {
            "global_baseline": {
                "description": "Baseline global calibration.",
                "test_metrics": {"roc_auc": 0.701, "brier_score": 0.214},
                "families": {
                    "PRIMARY": {"rows": 3, "evaluable_cells": 3, "di_pass": True, "eod_pass": True, "brier_pass": True, "di_min": 0.90, "eod_gap": 0.060, "brier_ratio": 1.08},
                    "SECONDARY": {"rows": 3, "evaluable_cells": 3, "di_pass": False, "eod_pass": True, "brier_pass": True, "di_min": 0.74, "eod_gap": 0.110, "brier_ratio": 1.12},
                },
            },
            "income_tertile_aware": {
                "description": "Income-tertile-specific calibrators with global fallback.",
                "test_metrics": {"roc_auc": 0.696, "brier_score": 0.209},
                "families": {
                    "PRIMARY": {"rows": 3, "evaluable_cells": 3, "di_pass": True, "eod_pass": True, "brier_pass": True, "di_min": 0.92, "eod_gap": 0.041, "brier_ratio": 1.04},
                    "SECONDARY": {"rows": 3, "evaluable_cells": 3, "di_pass": True, "eod_pass": True, "brier_pass": True, "di_min": 0.84, "eod_gap": 0.052, "brier_ratio": 1.03},
                },
            },
        },
    }
    (artifact_dir / "subgroup_calibration_experiment_report.json").write_text(
        json.dumps(report),
        encoding="utf-8",
    )

    snapshot = build_fairness_comparison_snapshot(artifact_dir, fairness_dir)

    assert snapshot["mode"] == "offline_comparison"
    assert snapshot["experimental"]["available"] is True
    assert snapshot["experimental"]["strategy_name"] == "income_tertile_aware"
    assert len(snapshot["experimental"]["tradeoff_notes"]) >= 2
