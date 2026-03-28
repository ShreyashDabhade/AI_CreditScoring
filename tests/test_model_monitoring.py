from __future__ import annotations

from pathlib import Path

from src.db_manager import create_application, init_database, save_score_run
from src.model_monitoring import compute_probability_drift_snapshot


def _seed_score_runs(tmp_path: Path, probabilities: list[float]) -> str:
    db_path = init_database(tmp_path / "monitoring.sqlite3")
    application = create_application(
        db_path,
        applicant_name="Model Monitor",
        tier_type="REDUCED",
        current_status="SUBMITTED",
        application_payload_json={"application": {"AMT_CREDIT": 180000}},
    )

    for probability in probabilities:
        save_score_run(
            db_path,
            application_id=application["id"],
            probability=probability,
            decision="REVIEW",
            model_version="test-model",
            score_payload_json={"probability_of_default": probability},
        )

    return db_path


def test_compute_probability_drift_snapshot_handles_low_sample_sizes(tmp_path: Path):
    db_path = _seed_score_runs(tmp_path, [0.19, 0.21, 0.2, 0.22, 0.18])

    snapshot = compute_probability_drift_snapshot(db_path, baseline_mean=0.21, min_sample_size=30)

    assert snapshot["sample_size"] == 5
    assert snapshot["status"] == "watch"
    assert snapshot["live_mean"] is not None
    assert "more live volume is needed" in snapshot["alert_message"].lower()


def test_compute_probability_drift_snapshot_flags_alert_for_material_deviation(tmp_path: Path):
    db_path = _seed_score_runs(tmp_path, [0.30] * 35)

    snapshot = compute_probability_drift_snapshot(db_path, baseline_mean=0.21, min_sample_size=30)

    assert snapshot["sample_size"] == 35
    assert snapshot["status"] == "alert"
    assert snapshot["recalibration_alert"] is True
    assert snapshot["deviation_pct"] is not None
    assert snapshot["deviation_pct"] > 10.0


def test_compute_probability_drift_snapshot_marks_stable_when_close_to_baseline(tmp_path: Path):
    db_path = _seed_score_runs(tmp_path, [0.209, 0.211, 0.212, 0.208] * 10)

    snapshot = compute_probability_drift_snapshot(db_path, baseline_mean=0.21, min_sample_size=30)

    assert snapshot["sample_size"] == 40
    assert snapshot["status"] == "stable"
    assert snapshot["recalibration_alert"] is False
    assert snapshot["live_std"] is not None
