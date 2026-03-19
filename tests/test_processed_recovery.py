import json
import warnings
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[1] / ".mplconfig"),
)

from src.builder_artifacts import (
    BuilderValidationError,
    builder_manifest_path,
    load_builder_artifact,
    load_processed_artifact_manifest,
    processed_manifest_fingerprint,
    validate_builder_pair,
)
from src.data_pipeline import build_adversarial_dataset, proxy_recency_sort
from src.fairness_audit import compute_fairness_metrics
from src.feature_engineering import (
    AGGREGATE_CONTRACT_VERSION,
    FEATURE_ENGINEERING_VERSION,
    fit_reduced_builder,
)
from tests.test_m2_m4_integration import _make_application_df


def test_proxy_recency_sort_uses_staleness_descending_with_median_fallback():
    df = pd.DataFrame(
        {
            "SK_ID_CURR": [10, 20, 30, 40],
            "DAYS_ID_PUBLISH": [np.nan, -400.0, np.nan, -100.0],
            "DAYS_REGISTRATION": [-1000.0, -200.0, -300.0, np.nan],
        }
    )

    result = proxy_recency_sort(df)

    assert result["SK_ID_CURR"].tolist() == [20, 10, 30, 40]


def test_build_adversarial_dataset_enforces_diagnostic_schema():
    train_df = pd.DataFrame(
        {
            "SK_ID_CURR": [1, 2, 3, 4],
            "TARGET": [0, 1, 0, 1],
            "AMT_INCOME_TOTAL": [100000.0, 120000.0, 140000.0, 160000.0],
            "AMT_INCOME_TOTAL_CAPPED": [100000.0, 120000.0, 140000.0, 150000.0],
            "DAYS_EMPLOYED": [-100.0, np.nan, -200.0, np.nan],
            "DAYS_EMPLOYED_ANOM": [0, 1, 0, 1],
        }
    )
    test_df = pd.DataFrame(
        {
            "SK_ID_CURR": [5, 6, 7, 8],
            "AMT_INCOME_TOTAL": [110000.0, 130000.0, 150000.0, 170000.0],
            "AMT_INCOME_TOTAL_CAPPED": [110000.0, 130000.0, 150000.0, 150000.0],
            "DAYS_EMPLOYED": [-150.0, np.nan, -250.0, np.nan],
            "DAYS_EMPLOYED_ANOM": [0, 1, 0, 1],
        }
    )

    adv_train, adv_val = build_adversarial_dataset(train_df, test_df)

    for df in [adv_train, adv_val]:
        assert "ADV_LABEL" in df.columns
        assert "TARGET" not in df.columns
        assert "AMT_INCOME_TOTAL_CAPPED" in df.columns
        assert "DAYS_EMPLOYED_ANOM" in df.columns
        assert df["SK_ID_CURR"].duplicated().sum() == 0

    assert [c for c in adv_train.columns if c != "ADV_LABEL"] == [
        c for c in adv_val.columns if c != "ADV_LABEL"
    ]


def test_processed_manifest_round_trip_rejects_bad_fingerprint(tmp_path):
    manifest = {
        "manifest_version": 1,
        "git_commit": "abc123",
        "application_train_cleaned_rows": 4,
        "income_cap": 123.0,
        "split_summary": {},
        "split_fingerprints": {},
        "split_schema_hashes": {},
        "adversarial_summary": {},
    }
    manifest["processed_manifest_fingerprint"] = processed_manifest_fingerprint(manifest)
    manifest_path = tmp_path / "processed_artifact_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    loaded = load_processed_artifact_manifest(str(manifest_path))
    assert loaded["processed_manifest_fingerprint"] == manifest["processed_manifest_fingerprint"]

    broken = dict(manifest)
    broken["processed_manifest_fingerprint"] = "broken"
    manifest_path.write_text(json.dumps(broken, indent=2), encoding="utf-8")

    with pytest.raises(BuilderValidationError):
        load_processed_artifact_manifest(str(manifest_path))


def test_builder_manifest_records_processed_manifest_fingerprint(tmp_path):
    train_df = _make_application_df([600001, 600002, 600003, 600004])
    builder_path = tmp_path / "reduced_feature_builder.joblib"
    processed_fp = "processed-fingerprint-123"

    fit_reduced_builder(
        train_df,
        artifact_path=str(builder_path),
        fit_split_name="train",
        strict_artifact_validation=False,
        processed_manifest_fingerprint=processed_fp,
    )

    manifest = json.loads(Path(builder_manifest_path(str(builder_path))).read_text(encoding="utf-8"))
    assert manifest["processed_manifest_fingerprint"] == processed_fp

    validate_builder_pair(
        str(builder_path),
        builder_manifest_path(str(builder_path)),
        fit_df=train_df,
        expected_tier="REDUCED",
        fit_split_name="train",
        feature_engineering_version=FEATURE_ENGINEERING_VERSION,
        aggregate_contract_version=AGGREGATE_CONTRACT_VERSION,
        expected_aggregate_feature_count=0,
        strict_validation=False,
        processed_manifest_fingerprint=processed_fp,
    )

    with pytest.raises(BuilderValidationError):
        validate_builder_pair(
            str(builder_path),
            builder_manifest_path(str(builder_path)),
            fit_df=train_df,
            expected_tier="REDUCED",
            fit_split_name="train",
            feature_engineering_version=FEATURE_ENGINEERING_VERSION,
            aggregate_contract_version=AGGREGATE_CONTRACT_VERSION,
            expected_aggregate_feature_count=0,
            strict_validation=False,
            processed_manifest_fingerprint="different-fingerprint",
        )


def test_strict_loader_rejects_quarantine_paths(tmp_path):
    train_df = _make_application_df([700001, 700002, 700003, 700004])
    builder_path = tmp_path / "quarantine" / "reduced_feature_builder.joblib"
    builder_path.parent.mkdir(parents=True, exist_ok=True)
    builder_path.write_bytes(b"placeholder")
    Path(builder_manifest_path(str(builder_path))).write_text("{}", encoding="utf-8")

    with pytest.raises(BuilderValidationError, match="quarantine"):
        load_builder_artifact(
            str(builder_path),
            fit_df=train_df,
            expected_tier="REDUCED",
            fit_split_name="train",
            feature_engineering_version=FEATURE_ENGINEERING_VERSION,
            aggregate_contract_version=AGGREGATE_CONTRACT_VERSION,
            expected_aggregate_feature_count=0,
        )


def test_compute_fairness_metrics_uses_boolean_pass_columns_without_future_warning():
    df = pd.DataFrame(
        {
            "FAIR_GROUP_PRIMARY": ["A", "A", "B", "B"],
            "TARGET": [0, 1, 0, 1],
            "CALIBRATED_PD": [0.10, 0.90, 0.20, 0.80],
            "DECISION": ["APPROVE", "DECLINE", "APPROVE", "DECLINE"],
        }
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = compute_fairness_metrics(
            df,
            "FAIR_GROUP_PRIMARY",
            min_n=1,
            min_pos=1,
        )

    assert not any(isinstance(item.message, FutureWarning) for item in caught)
    assert str(result["di_pass"].dtype) == "boolean"
    assert str(result["eod_pass"].dtype) == "boolean"
    assert str(result["brier_pass"].dtype) == "boolean"
