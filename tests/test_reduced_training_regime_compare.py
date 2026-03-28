import json
from types import SimpleNamespace

import pandas as pd


def test_compare_metric_deltas_reports_directional_changes():
    from src.models.reduced_training_regime_compare import _compare_metric_deltas

    baseline = {"roc_auc": 0.70, "brier_score": 0.10, "accuracy": 0.80}
    candidate = {"roc_auc": 0.72, "brier_score": 0.09, "accuracy": 0.79}

    delta = _compare_metric_deltas(baseline, candidate)

    assert delta["test_auc_delta"] > 0.0
    assert delta["test_brier_delta"] < 0.0
    assert delta["test_accuracy_delta"] < 0.0


def test_run_reduced_training_regime_comparison_persists_report(tmp_path, monkeypatch):
    from src.models import reduced_training_regime_compare as rtc

    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    bundle = SimpleNamespace(
        mode="real",
        train=pd.DataFrame({"TARGET": [0, 1]}),
        val_model=pd.DataFrame({"TARGET": [0]}),
        val_policy=pd.DataFrame({"TARGET": [1]}),
        test=pd.DataFrame({"TARGET": [0, 1]}),
    )

    monkeypatch.setattr(rtc, "_has_real_training_inputs", lambda *args, **kwargs: True)
    monkeypatch.setattr(rtc, "_load_real_bundle", lambda *args, **kwargs: bundle)
    monkeypatch.setattr(
        rtc,
        "_runtime_reduced_summary",
        lambda *args, **kwargs: {
            "model_version": "reduced_v2.1.0",
            "test_metrics": {"accuracy": 0.80, "roc_auc": 0.70, "brier_score": 0.10, "default_rate": 0.50},
        },
    )

    def _stub_train(bundle, *, processed_dir, training_regime):
        if training_regime == "train_only":
            return {
                "regime_details": {"training_regime": training_regime},
                "sample_counts": {"train": 2, "val_model": 1, "calibration_holdout": 1, "test": 2},
                "selected_candidate": "baseline",
                "selected_params": {"max_depth": 4},
                "selected_calibrator": "isotonic",
                "val_model_roc_auc": 0.71,
                "val_policy_calibration_metrics": {"roc_auc": 0.70, "brier_score": 0.10},
                "test_metrics": {"accuracy": 0.81, "roc_auc": 0.71, "brier_score": 0.095, "default_rate": 0.50},
            }
        return {
            "regime_details": {"training_regime": training_regime},
            "sample_counts": {"train": 3, "val_model": 1, "calibration_holdout": 1, "test": 2},
            "selected_candidate": "baseline",
            "selected_params": {"max_depth": 4},
            "selected_calibrator": "isotonic",
            "val_model_roc_auc": 0.72,
            "val_policy_calibration_metrics": {"roc_auc": 0.71, "brier_score": 0.094},
            "test_metrics": {"accuracy": 0.82, "roc_auc": 0.73, "brier_score": 0.090, "default_rate": 0.50},
        }

    monkeypatch.setattr(rtc, "_train_reduced_regime", _stub_train)
    monkeypatch.setattr(rtc, "_processed_manifest_fingerprint", lambda *args, **kwargs: "processed-xyz")

    report = rtc.run_reduced_training_regime_comparison(
        artifact_dir=str(artifact_dir),
        processed_dir=str(processed_dir),
        raw_dir=str(raw_dir),
    )

    report_path = artifact_dir / rtc.REDUCED_TRAINING_REGIME_COMPARE_REPORT_FILENAME
    assert report_path.exists()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["training_regimes"]["train_only"]["test_metrics"]["roc_auc"] == 0.71
    assert payload["training_regimes"]["train_plus_val_policy"]["test_metrics"]["roc_auc"] == 0.73
    assert payload["delta_merged_vs_train_only"]["test_auc_delta"] > 0.0
    assert report["report_path"] == str(report_path)
