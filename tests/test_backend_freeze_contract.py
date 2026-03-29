from __future__ import annotations

import shutil
from pathlib import Path

from src.api.app import _build_demo_seed_payload, create_app
from src.builder_artifacts import load_validated_builders


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PROCESSED_DIR = REPO_ROOT / "data" / "processed"
RUNTIME_ARTIFACT_DIR = REPO_ROOT / "artifacts"
REQUIRED_RUNTIME_FILES = (
    "full_feature_builder.joblib",
    "full_feature_builder.manifest.json",
    "reduced_feature_builder.joblib",
    "reduced_feature_builder.manifest.json",
    "full_model.joblib",
    "full_calibrator.joblib",
    "full_shap_explainer.joblib",
    "reduced_model.joblib",
    "reduced_calibrator.joblib",
    "reduced_shap_explainer.joblib",
    "model_fairness_audit_passed.joblib",
    "reproducibility_report.json",
)
OFFLINE_ONLY_DIRS = (
    "reduced_thin_blend",
    "reduced_thin_lgbm",
    "reduced_thin_diag",
    "reduced_thin_opt",
    "alt_stacked_reduced",
    "random_stratified_reduced",
    "random_stratified_full_and_reduced",
)


def test_canonical_runtime_files_exist_for_real_mode_startup():
    assert (RUNTIME_PROCESSED_DIR / "processed_artifact_manifest.json").exists()
    for name in REQUIRED_RUNTIME_FILES:
        assert (RUNTIME_ARTIFACT_DIR / name).exists(), name


def test_canonical_builders_load_and_validate_for_both_runtime_tiers():
    builders = load_validated_builders(
        artifact_dir=str(RUNTIME_ARTIFACT_DIR),
        processed_dir=str(RUNTIME_PROCESSED_DIR),
        strict_artifacts=True,
    )

    assert set(builders) == {"FULL", "REDUCED"}
    assert builders["FULL"].tier == "FULL"
    assert builders["REDUCED"].tier == "REDUCED"
    assert len(builders["FULL"].encoded_columns_) > 0
    assert len(builders["REDUCED"].encoded_columns_) > 0


def test_real_runtime_default_stack_serves_frozen_health_and_score_contract():
    app = create_app()
    client = app.test_client()

    health = client.get("/health")
    assert health.status_code == 200
    health_payload = health.get_json()
    assert set(health_payload) == {
        "status",
        "model_version",
        "fairness_audit_passed",
        "coverage_tiers_available",
    }
    assert health_payload["status"] == "ok"
    assert health_payload["coverage_tiers_available"] == ["FULL", "REDUCED"]
    assert isinstance(health_payload["model_version"], str)
    assert isinstance(health_payload["fairness_audit_passed"], bool)

    sample_payload = _build_demo_seed_payload()

    reduced = client.post("/score", json={"application": sample_payload["application"]})
    assert reduced.status_code == 200
    reduced_payload = reduced.get_json()
    assert set(reduced_payload) == {
        "probability_of_default",
        "decision",
        "escalate",
        "top_5_explanations",
        "model_version",
        "calibrated",
        "model_fairness_audit_passed",
        "fairness_audit_version",
        "coverage_tier",
    }
    assert reduced_payload["coverage_tier"] == "REDUCED"
    assert reduced_payload["decision"] in {"APPROVE", "REVIEW", "DECLINE"}
    assert isinstance(reduced_payload["probability_of_default"], float)
    assert isinstance(reduced_payload["calibrated"], bool)
    assert isinstance(reduced_payload["model_fairness_audit_passed"], bool)
    assert len(reduced_payload["top_5_explanations"]) == 5

    full = client.post("/score", json=sample_payload)
    assert full.status_code == 200
    full_payload = full.get_json()
    assert full_payload["coverage_tier"] == "FULL"
    assert full_payload["decision"] in {"APPROVE", "REVIEW", "DECLINE"}
    assert len(full_payload["top_5_explanations"]) == 5


def test_offline_experiment_directories_are_not_runtime_api_inputs(tmp_path):
    artifact_dir = tmp_path / "artifacts"
    processed_dir = tmp_path / "processed"
    artifact_dir.mkdir()
    processed_dir.mkdir()

    shutil.copy2(
        RUNTIME_PROCESSED_DIR / "processed_artifact_manifest.json",
        processed_dir / "processed_artifact_manifest.json",
    )
    for name in REQUIRED_RUNTIME_FILES:
        shutil.copy2(RUNTIME_ARTIFACT_DIR / name, artifact_dir / name)

    for dirname in OFFLINE_ONLY_DIRS:
        offline_dir = artifact_dir / dirname
        offline_dir.mkdir()
        (offline_dir / "sentinel.txt").write_text("offline-only", encoding="utf-8")

    app = create_app(
        artifact_dir=str(artifact_dir),
        processed_dir=str(processed_dir),
        mock_mode=False,
        strict_artifacts=True,
    )
    response = app.test_client().get("/health")

    assert response.status_code == 200
    assert response.get_json()["coverage_tiers_available"] == ["FULL", "REDUCED"]
