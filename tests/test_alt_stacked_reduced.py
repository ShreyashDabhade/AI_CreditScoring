from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest


def _write_csv(tmp_path, filename: str, df: pd.DataFrame):
    path = tmp_path / filename
    df.to_csv(path, index=False)
    return path


def _make_telco_df(sk_ids: list[int]) -> pd.DataFrame:
    rows = []
    for offset, sk_id in enumerate(sk_ids, start=1):
        rows.append(
            {
                "SK_ID_CURR": sk_id,
                "TELECOM_LATE_DAYS_MAX_6M": 1.0 * offset,
                "UTILITY_DISCONNECT_FLAGS_12M": offset % 2,
                "PREPAID_TOP_UP_VOLATILITY": 0.1 * offset,
                "TELECOM_ON_TIME_PAYMENT_RATE_6M": 0.9,
                "UTILITY_LATE_PAYMENT_COUNT_12M": offset,
                "AVG_MONTHLY_TOPUP_AMOUNT": 100.0 + offset,
                "AVG_UTILITY_BILL_AMOUNT": 250.0 + offset,
                "TELCO_PAYMENT_REGULARITY_SCORE": 0.7,
                "UTILITY_STRESS_SCORE": 0.2,
                "DIGITAL_SERVICE_STABILITY_SCORE": 0.8,
            }
        )
    return pd.DataFrame(rows)


def _make_wallet_df(sk_ids: list[int]) -> pd.DataFrame:
    rows = []
    for offset, sk_id in enumerate(sk_ids, start=1):
        rows.append(
            {
                "SK_ID_CURR": sk_id,
                "P2P_MICRO_INFLOW_COUNT_30D": offset,
                "WALLET_BALANCE_DEPLETION_RATE": 0.1 * offset,
                "MERCHANT_PAYMENT_RATIO": 0.6,
                "P2P_OUTFLOW_COUNT_30D": offset + 1,
                "DIGITAL_WALLET_CASHIN_COUNT_30D": offset + 2,
                "DIGITAL_WALLET_CASHOUT_COUNT_30D": offset + 3,
                "AVG_WALLET_TXN_AMOUNT": 50.0 + offset,
                "WALLET_TXN_VOLATILITY": 0.3,
                "WALLET_LIQUIDITY_STRESS_SCORE": 0.25,
                "PEER_BORROWING_SIGNAL_SCORE": 0.4,
                "FORMAL_MERCHANT_STABILITY_SCORE": 0.9,
            }
        )
    return pd.DataFrame(rows)


def _make_ecommerce_df(sk_ids: list[int]) -> pd.DataFrame:
    rows = []
    for offset, sk_id in enumerate(sk_ids, start=1):
        rows.append(
            {
                "SK_ID_CURR": sk_id,
                "CASH_ON_DELIVERY_RATIO": 0.2,
                "SOCIAL_ACCOUNT_AGE_MONTHS": 12 + offset,
                "HIGH_RISK_MERCHANT_TXNS": offset,
                "ECOMMERCE_ORDER_COUNT_6M": 20 + offset,
                "ECOMMERCE_RETURN_RATE": 0.05,
                "AVG_BASKET_VALUE": 500.0 + offset,
                "MERCHANT_CATEGORY_DIVERSITY": 4 + offset,
                "SOCIAL_ENGAGEMENT_STABILITY_SCORE": 0.6,
                "DIGITAL_TRUST_SCORE": 0.7,
                "FRAUD_RISK_PROXY_SCORE": 0.1,
            }
        )
    return pd.DataFrame(rows)


def test_missing_required_column_raises_clear_error(tmp_path):
    from src.models.alt_stacked_reduced_sidecars import load_telco_sidecar

    df = _make_telco_df([101, 102]).drop(columns=["AVG_UTILITY_BILL_AMOUNT"])
    path = _write_csv(tmp_path, "telco_missing.csv", df)

    with pytest.raises(ValueError, match="telco_utility sidecar is missing required columns"):
        load_telco_sidecar(path)


def test_duplicate_sk_id_curr_raises_clear_error(tmp_path):
    from src.models.alt_stacked_reduced_sidecars import load_wallet_sidecar

    df = _make_wallet_df([201, 201, 202])
    path = _write_csv(tmp_path, "wallet_duplicates.csv", df)

    with pytest.raises(ValueError, match="wallet_p2p sidecar contains duplicate SK_ID_CURR values"):
        load_wallet_sidecar(path)


def test_telco_alignment_preserves_master_row_order():
    from src.models.alt_stacked_reduced_sidecars import align_sidecar_to_ids

    master = pd.DataFrame({"SK_ID_CURR": [3, 1, 2]})
    sidecar = _make_telco_df([1, 2, 3])

    aligned, metadata = align_sidecar_to_ids(master, sidecar, "telco_utility")

    assert aligned["SK_ID_CURR"].tolist() == [3, 1, 2]
    assert aligned["TELECOM_LATE_DAYS_MAX_6M"].tolist() == [3.0, 1.0, 2.0]
    assert metadata["matched_row_count"] == 3


def test_wallet_alignment_preserves_master_row_order():
    from src.models.alt_stacked_reduced_sidecars import align_sidecar_to_ids

    master = pd.DataFrame({"SK_ID_CURR": [30, 10, 20]})
    sidecar = _make_wallet_df([10, 20, 30])

    aligned, metadata = align_sidecar_to_ids(master, sidecar, "wallet_p2p")

    assert aligned["SK_ID_CURR"].tolist() == [30, 10, 20]
    assert aligned["P2P_MICRO_INFLOW_COUNT_30D"].tolist() == [3, 1, 2]
    assert metadata["matched_row_count"] == 3


def test_ecommerce_alignment_preserves_master_row_order():
    from src.models.alt_stacked_reduced_sidecars import align_sidecar_to_ids

    master = pd.DataFrame({"SK_ID_CURR": [300, 100, 200]})
    sidecar = _make_ecommerce_df([100, 200, 300])

    aligned, metadata = align_sidecar_to_ids(master, sidecar, "ecommerce_social")

    assert aligned["SK_ID_CURR"].tolist() == [300, 100, 200]
    assert aligned["SOCIAL_ACCOUNT_AGE_MONTHS"].tolist() == [15, 13, 14]
    assert metadata["matched_row_count"] == 3


def test_parent_source_file_is_accepted_but_excluded_from_feature_columns(tmp_path):
    from src.models.alt_stacked_reduced_sidecars import (
        PARENT_SOURCE_FILE_COL,
        extract_sidecar_feature_columns,
        load_telco_sidecar,
    )

    df = _make_telco_df([501, 502])
    df[PARENT_SOURCE_FILE_COL] = ["a.csv", "b.csv"]
    path = _write_csv(tmp_path, "telco_parent_source.csv", df)

    loaded = load_telco_sidecar(path)
    feature_columns = extract_sidecar_feature_columns(loaded)

    assert PARENT_SOURCE_FILE_COL in loaded.columns
    assert PARENT_SOURCE_FILE_COL not in feature_columns
    assert "SK_ID_CURR" not in feature_columns


def test_unmatched_sk_id_curr_rows_are_reported_explicitly():
    from src.models.alt_stacked_reduced_sidecars import align_sidecar_to_ids

    master = pd.DataFrame({"SK_ID_CURR": [700, 701, 702]})
    sidecar = _make_wallet_df([700, 702, 999])

    aligned, metadata = align_sidecar_to_ids(master, sidecar, "wallet_p2p")

    assert aligned["SK_ID_CURR"].tolist() == [700, 701, 702]
    assert pd.isna(aligned.loc[1, "AVG_WALLET_TXN_AMOUNT"])
    assert metadata["unmatched_master_ids"] == [701]
    assert metadata["unused_sidecar_ids"] == [999]
    assert metadata["unmatched_master_row_count"] == 1
    assert metadata["has_missing_coverage"] is True


def test_feature_column_extractor_returns_only_model_usable_sidecar_features():
    from src.models.alt_stacked_reduced_sidecars import (
        ECOMMERCE_SOCIAL_REQUIRED_COLUMNS,
        PARENT_SOURCE_FILE_COL,
        extract_sidecar_feature_columns,
    )

    df = _make_ecommerce_df([801])
    df[PARENT_SOURCE_FILE_COL] = ["source.csv"]

    feature_columns = extract_sidecar_feature_columns(df)

    expected = [column for column in ECOMMERCE_SOCIAL_REQUIRED_COLUMNS if column != "SK_ID_CURR"]
    assert feature_columns == expected


class _RecordingTelcoModel:
    fit_feature_values_log: list[list[float]] = []

    def fit(self, X, y, verbose=False):
        feature_values = X["TELECOM_LATE_DAYS_MAX_6M"].astype(float).tolist()
        type(self).fit_feature_values_log.append(feature_values)
        self._probability_floor = float(np.mean(y)) if len(y) else 0.5
        return self

    def predict_proba(self, X):
        values = X["TELECOM_LATE_DAYS_MAX_6M"].to_numpy(dtype=float)
        centered = values - float(np.mean(values))
        probs = 1.0 / (1.0 + np.exp(-centered / 10.0))
        probs = np.clip(probs, 0.001, 0.999)
        return np.column_stack([1.0 - probs, probs])


class _RecordingWalletModel:
    fit_feature_values_log: list[list[float]] = []

    def fit(self, X, y, verbose=False):
        feature_values = X["P2P_MICRO_INFLOW_COUNT_30D"].astype(float).tolist()
        type(self).fit_feature_values_log.append(feature_values)
        return self

    def predict_proba(self, X):
        values = X["P2P_MICRO_INFLOW_COUNT_30D"].to_numpy(dtype=float)
        centered = values - float(np.mean(values))
        probs = 1.0 / (1.0 + np.exp(-centered / 2.0))
        probs = np.clip(probs, 0.001, 0.999)
        return np.column_stack([1.0 - probs, probs])


class _RecordingEcommerceModel:
    fit_feature_values_log: list[list[float]] = []

    def fit(self, X, y, verbose=False):
        feature_values = X["ECOMMERCE_ORDER_COUNT_6M"].astype(float).tolist()
        type(self).fit_feature_values_log.append(feature_values)
        return self

    def predict_proba(self, X):
        values = X["ECOMMERCE_ORDER_COUNT_6M"].to_numpy(dtype=float)
        centered = values - float(np.mean(values))
        probs = 1.0 / (1.0 + np.exp(-centered / 3.0))
        probs = np.clip(probs, 0.001, 0.999)
        return np.column_stack([1.0 - probs, probs])


class _FakeReducedBuilder:
    pass


class _RecordingMasterModel:
    fit_row_counts: list[int] = []
    fit_columns: list[list[str]] = []
    predict_row_counts: list[int] = []

    def fit(self, X, y, verbose=False):
        type(self).fit_row_counts.append(len(X))
        type(self).fit_columns.append(list(X.columns))
        return self

    def predict_proba(self, X):
        type(self).predict_row_counts.append(len(X))
        signal = X["META_SCORE_TELCO"].to_numpy(dtype=float) + X["META_SCORE_WALLET"].to_numpy(dtype=float)
        signal = signal - float(np.mean(signal))
        probs = 1.0 / (1.0 + np.exp(-signal))
        probs = np.clip(probs, 0.001, 0.999)
        return np.column_stack([1.0 - probs, probs])


class _RecordingMasterCalibrator:
    predict_call_row_counts: list[int] = []

    def predict(self, scores):
        arr = np.asarray(scores, dtype=float)
        type(self).predict_call_row_counts.append(len(arr))
        return np.clip(arr * 0.95 + 0.01, 0.001, 0.999)


class _RecordingAblationModel:
    fit_columns: list[list[str]] = []
    fit_row_counts: list[int] = []

    def fit(self, X, y, verbose=False):
        type(self).fit_columns.append(list(X.columns))
        type(self).fit_row_counts.append(len(X))
        return self

    def predict_proba(self, X):
        signal = np.zeros(len(X), dtype=float)
        for column in X.columns:
            signal += X[column].to_numpy(dtype=float)
        signal = signal - float(np.mean(signal))
        probs = 1.0 / (1.0 + np.exp(-signal / max(1.0, X.shape[1])))
        probs = np.clip(probs, 0.001, 0.999)
        return np.column_stack([1.0 - probs, probs])


def _make_telco_bundle():
    def _frame(sk_ids: list[int], targets: list[int]) -> pd.DataFrame:
        return pd.DataFrame({"SK_ID_CURR": sk_ids, "TARGET": targets})

    train_ids = list(range(1001, 1021))
    val_model_ids = list(range(2001, 2007))
    val_policy_ids = list(range(3001, 3007))
    test_ids = list(range(4001, 4009))

    train_targets = [0, 1] * 10
    val_model_targets = [0, 1, 0, 1, 0, 1]
    val_policy_targets = [1, 0, 1, 0, 1, 0]
    test_targets = [0, 1, 0, 1, 0, 1, 0, 1]

    return {
        "train_ids": train_ids,
        "val_model_ids": val_model_ids,
        "val_policy_ids": val_policy_ids,
        "test_ids": test_ids,
        "bundle": type(
            "Bundle",
            (),
            {
                "mode": "real",
                "train": _frame(train_ids, train_targets),
                "val_model": _frame(val_model_ids, val_model_targets),
                "val_policy": _frame(val_policy_ids, val_policy_targets),
                "test": _frame(test_ids, test_targets),
                "raw_dir": "unused",
                "uses_flattened_full_input": False,
            },
        )(),
    }


def _make_telco_sidecar_for_bundle(bundle_payload: dict[str, object]) -> pd.DataFrame:
    all_ids = (
        bundle_payload["train_ids"]
        + bundle_payload["val_model_ids"]
        + bundle_payload["val_policy_ids"]
        + bundle_payload["test_ids"]
    )
    df = _make_telco_df(all_ids)  # type: ignore[arg-type]
    df["TELECOM_LATE_DAYS_MAX_6M"] = df["SK_ID_CURR"].astype(float)
    df["UTILITY_DISCONNECT_FLAGS_12M"] = (df["SK_ID_CURR"] % 2).astype(int)
    df["PREPAID_TOP_UP_VOLATILITY"] = df["SK_ID_CURR"].astype(float) / 10000.0
    df["TELECOM_ON_TIME_PAYMENT_RATE_6M"] = 0.5 + (df["SK_ID_CURR"] % 7) / 20.0
    df["UTILITY_LATE_PAYMENT_COUNT_12M"] = (df["SK_ID_CURR"] % 5).astype(int)
    df["AVG_MONTHLY_TOPUP_AMOUNT"] = df["SK_ID_CURR"].astype(float) / 10.0
    df["AVG_UTILITY_BILL_AMOUNT"] = df["SK_ID_CURR"].astype(float) / 8.0
    df["TELCO_PAYMENT_REGULARITY_SCORE"] = 0.4 + (df["SK_ID_CURR"] % 11) / 20.0
    df["UTILITY_STRESS_SCORE"] = (df["SK_ID_CURR"] % 13) / 20.0
    df["DIGITAL_SERVICE_STABILITY_SCORE"] = 0.3 + (df["SK_ID_CURR"] % 17) / 20.0
    return df


def _make_wallet_sidecar_for_bundle(bundle_payload: dict[str, object]) -> pd.DataFrame:
    all_ids = (
        bundle_payload["train_ids"]
        + bundle_payload["val_model_ids"]
        + bundle_payload["val_policy_ids"]
        + bundle_payload["test_ids"]
    )
    df = _make_wallet_df(all_ids)  # type: ignore[arg-type]
    df["P2P_MICRO_INFLOW_COUNT_30D"] = df["SK_ID_CURR"].astype(float)
    df["WALLET_BALANCE_DEPLETION_RATE"] = (df["SK_ID_CURR"] % 7).astype(float) / 10.0
    df["MERCHANT_PAYMENT_RATIO"] = 0.3 + (df["SK_ID_CURR"] % 5) / 10.0
    df["P2P_OUTFLOW_COUNT_30D"] = (df["SK_ID_CURR"] % 11).astype(float) + 1.0
    df["DIGITAL_WALLET_CASHIN_COUNT_30D"] = (df["SK_ID_CURR"] % 13).astype(float) + 1.0
    df["DIGITAL_WALLET_CASHOUT_COUNT_30D"] = (df["SK_ID_CURR"] % 6).astype(float) + 1.0
    df["AVG_WALLET_TXN_AMOUNT"] = df["SK_ID_CURR"].astype(float) / 12.0
    df["WALLET_TXN_VOLATILITY"] = (df["SK_ID_CURR"] % 8).astype(float) / 10.0
    df["WALLET_LIQUIDITY_STRESS_SCORE"] = (df["SK_ID_CURR"] % 10).astype(float) / 10.0
    df["PEER_BORROWING_SIGNAL_SCORE"] = 0.2 + (df["SK_ID_CURR"] % 7) / 10.0
    df["FORMAL_MERCHANT_STABILITY_SCORE"] = 0.4 + (df["SK_ID_CURR"] % 9) / 10.0
    return df


def _make_ecommerce_sidecar_for_bundle(bundle_payload: dict[str, object]) -> pd.DataFrame:
    all_ids = (
        bundle_payload["train_ids"]
        + bundle_payload["val_model_ids"]
        + bundle_payload["val_policy_ids"]
        + bundle_payload["test_ids"]
    )
    df = _make_ecommerce_df(all_ids)  # type: ignore[arg-type]
    df["CASH_ON_DELIVERY_RATIO"] = (df["SK_ID_CURR"] % 5).astype(float) / 10.0
    df["SOCIAL_ACCOUNT_AGE_MONTHS"] = (df["SK_ID_CURR"] % 24).astype(float) + 1.0
    df["HIGH_RISK_MERCHANT_TXNS"] = (df["SK_ID_CURR"] % 7).astype(float)
    df["ECOMMERCE_ORDER_COUNT_6M"] = df["SK_ID_CURR"].astype(float)
    df["ECOMMERCE_RETURN_RATE"] = (df["SK_ID_CURR"] % 6).astype(float) / 20.0
    df["AVG_BASKET_VALUE"] = df["SK_ID_CURR"].astype(float) / 9.0
    df["MERCHANT_CATEGORY_DIVERSITY"] = (df["SK_ID_CURR"] % 8).astype(float) + 1.0
    df["SOCIAL_ENGAGEMENT_STABILITY_SCORE"] = 0.2 + (df["SK_ID_CURR"] % 10) / 10.0
    df["DIGITAL_TRUST_SCORE"] = 0.3 + (df["SK_ID_CURR"] % 11) / 10.0
    df["FRAUD_RISK_PROXY_SCORE"] = (df["SK_ID_CURR"] % 9).astype(float) / 10.0
    return df


def _run_telco_experiment(tmp_path, monkeypatch):
    from src.models import alt_stacked_reduced as asr

    bundle_payload = _make_telco_bundle()
    sidecar_path = _write_csv(
        tmp_path,
        "telco_sidecar.csv",
        _make_telco_sidecar_for_bundle(bundle_payload),
    )

    _RecordingTelcoModel.fit_feature_values_log = []
    monkeypatch.setattr(asr, "load_alt_stacked_reduced_bundle", lambda **kwargs: bundle_payload["bundle"])
    monkeypatch.setattr(asr, "_candidate_model_params", lambda scale_pos_weight: [("stub", {})])
    monkeypatch.setattr(asr, "_make_model", lambda params: _RecordingTelcoModel())

    artifact_root = tmp_path / "artifacts"
    report = asr.run_telco_submodel_experiment(
        telco_sidecar_path=sidecar_path,
        artifact_dir=str(artifact_root),
        processed_dir=str(tmp_path / "processed"),
        raw_dir=str(tmp_path / "raw"),
        n_oof_folds=5,
    )
    return report, bundle_payload, artifact_root / "alt_stacked_reduced"


def _run_wallet_experiment(tmp_path, monkeypatch):
    from src.models import alt_stacked_reduced as asr

    bundle_payload = _make_telco_bundle()
    sidecar_path = _write_csv(
        tmp_path,
        "wallet_sidecar.csv",
        _make_wallet_sidecar_for_bundle(bundle_payload),
    )

    _RecordingWalletModel.fit_feature_values_log = []
    monkeypatch.setattr(asr, "load_alt_stacked_reduced_bundle", lambda **kwargs: bundle_payload["bundle"])
    monkeypatch.setattr(asr, "_candidate_model_params", lambda scale_pos_weight: [("stub", {})])
    monkeypatch.setattr(asr, "_make_model", lambda params: _RecordingWalletModel())

    artifact_root = tmp_path / "artifacts"
    report = asr.run_wallet_submodel_experiment(
        wallet_sidecar_path=sidecar_path,
        artifact_dir=str(artifact_root),
        processed_dir=str(tmp_path / "processed"),
        raw_dir=str(tmp_path / "raw"),
        n_oof_folds=5,
    )
    return report, bundle_payload, artifact_root / "alt_stacked_reduced"


def _run_ecommerce_experiment(tmp_path, monkeypatch):
    from src.models import alt_stacked_reduced as asr

    bundle_payload = _make_telco_bundle()
    sidecar_path = _write_csv(
        tmp_path,
        "ecommerce_sidecar.csv",
        _make_ecommerce_sidecar_for_bundle(bundle_payload),
    )

    _RecordingEcommerceModel.fit_feature_values_log = []
    monkeypatch.setattr(asr, "load_alt_stacked_reduced_bundle", lambda **kwargs: bundle_payload["bundle"])
    monkeypatch.setattr(asr, "_candidate_model_params", lambda scale_pos_weight: [("stub", {})])
    monkeypatch.setattr(asr, "_make_model", lambda params: _RecordingEcommerceModel())

    artifact_root = tmp_path / "artifacts"
    report = asr.run_ecommerce_submodel_experiment(
        ecommerce_sidecar_path=sidecar_path,
        artifact_dir=str(artifact_root),
        processed_dir=str(tmp_path / "processed"),
        raw_dir=str(tmp_path / "raw"),
        n_oof_folds=5,
    )
    return report, bundle_payload, artifact_root / "alt_stacked_reduced"


def _run_meta_assembly_experiment(tmp_path, monkeypatch, track_order: bool = False):
    from src.models import alt_stacked_reduced as asr

    bundle_payload = _make_telco_bundle()
    artifact_root = tmp_path / "artifacts"
    processed_root = tmp_path / "processed"

    _run_telco_experiment(tmp_path, monkeypatch)
    _run_wallet_experiment(tmp_path, monkeypatch)
    _run_ecommerce_experiment(tmp_path, monkeypatch)

    call_order: list[str] = []
    monkeypatch.setattr(asr, "load_alt_stacked_reduced_bundle", lambda **kwargs: bundle_payload["bundle"])
    monkeypatch.setattr(asr, "fit_reduced_builder", lambda *args, **kwargs: _FakeReducedBuilder())

    def _fake_build_reduced(df, builder, for_linear_model=False):
        if track_order:
            call_order.append("build_reduced")
        n = len(df)
        return pd.DataFrame(
            {
                "canonical_reduced_a": np.arange(n, dtype=float),
                "canonical_reduced_b": df["SK_ID_CURR"].to_numpy(dtype=float),
            }
        )

    monkeypatch.setattr(asr, "build_reduced", _fake_build_reduced)

    if track_order:
        original_append = asr.append_meta_scores_to_reduced_matrix

        def _recording_append(reduced_matrix, meta_scores_df):
            call_order.append("append_meta")
            return original_append(reduced_matrix, meta_scores_df)

        monkeypatch.setattr(asr, "append_meta_scores_to_reduced_matrix", _recording_append)

    summary = asr.run_meta_score_assembly_experiment(
        artifact_dir=str(artifact_root),
        processed_dir=str(processed_root),
        raw_dir=str(tmp_path / "raw"),
    )
    return summary, bundle_payload, artifact_root / "alt_stacked_reduced", processed_root / "alt_stacked_reduced", call_order


def _run_master_xgb_experiment(tmp_path, monkeypatch):
    from src.models import alt_stacked_reduced as asr

    _, bundle_payload, artifact_dir, processed_dir, _ = _run_meta_assembly_experiment(tmp_path, monkeypatch)

    _RecordingMasterModel.fit_row_counts = []
    _RecordingMasterModel.fit_columns = []
    _RecordingMasterModel.predict_row_counts = []
    _RecordingMasterCalibrator.predict_call_row_counts = []

    monkeypatch.setattr(asr, "load_alt_stacked_reduced_bundle", lambda **kwargs: bundle_payload["bundle"])
    monkeypatch.setattr(asr, "_candidate_model_params", lambda scale_pos_weight: [("stub", {})])
    monkeypatch.setattr(asr, "_make_model", lambda params: _RecordingMasterModel())

    calibration_calls: list[dict[str, int]] = []

    def _fake_select_calibrator(y_true, raw_scores):
        calibration_calls.append({"y_len": len(y_true), "raw_len": len(raw_scores)})
        return _RecordingMasterCalibrator(), "stub_calibrator", {"roc_auc": 0.75, "brier_score": 0.10}

    monkeypatch.setattr(asr, "_select_calibrator", _fake_select_calibrator)
    monkeypatch.setattr(
        asr,
        "load_frozen_reduced_baseline_metrics",
        lambda *args, **kwargs: {
            "source": "stubbed_frozen_reduced",
            "val_model": {
                "raw": {"roc_auc": 0.70, "pr_auc": 0.20, "brier_score": 0.12, "default_rate": 0.5},
                "calibrated": {"roc_auc": 0.71, "pr_auc": 0.21, "brier_score": 0.11, "default_rate": 0.5},
            },
            "val_policy": {
                "raw": {"roc_auc": 0.69, "pr_auc": 0.19, "brier_score": 0.13, "default_rate": 0.5},
                "calibrated": {"roc_auc": 0.70, "pr_auc": 0.20, "brier_score": 0.12, "default_rate": 0.5},
            },
            "test": {
                "raw": {"roc_auc": 0.68, "pr_auc": 0.18, "brier_score": 0.14, "default_rate": 0.5},
                "calibrated": {"roc_auc": 0.69, "pr_auc": 0.19, "brier_score": 0.13, "default_rate": 0.5},
            },
        },
    )

    report = asr.run_alt_stacked_reduced_master_xgb_experiment(
        artifact_dir=str(tmp_path / "artifacts"),
        processed_dir=str(tmp_path / "processed"),
        raw_dir=str(tmp_path / "raw"),
    )
    return report, bundle_payload, artifact_dir, processed_dir, calibration_calls


def _run_ablation_experiment(tmp_path, monkeypatch):
    from src.models import alt_stacked_reduced as asr

    _, bundle_payload, artifact_dir, processed_dir, _ = _run_meta_assembly_experiment(tmp_path, monkeypatch)

    _RecordingAblationModel.fit_columns = []
    _RecordingAblationModel.fit_row_counts = []
    _RecordingMasterCalibrator.predict_call_row_counts = []

    monkeypatch.setattr(asr, "load_alt_stacked_reduced_bundle", lambda **kwargs: bundle_payload["bundle"])
    monkeypatch.setattr(asr, "_candidate_model_params", lambda scale_pos_weight: [("stub", {})])
    monkeypatch.setattr(asr, "_make_model", lambda params: _RecordingAblationModel())

    calibration_calls: list[dict[str, int]] = []

    def _fake_select_calibrator(y_true, raw_scores):
        calibration_calls.append({"y_len": len(y_true), "raw_len": len(raw_scores)})
        return _RecordingMasterCalibrator(), "stub_calibrator", {"roc_auc": 0.74, "brier_score": 0.09}

    monkeypatch.setattr(asr, "_select_calibrator", _fake_select_calibrator)
    monkeypatch.setattr(
        asr,
        "load_frozen_reduced_baseline_metrics",
        lambda *args, **kwargs: {
            "source": "stubbed_frozen_reduced",
            "val_model": {
                "raw": {"roc_auc": 0.70, "pr_auc": 0.20, "brier_score": 0.12, "default_rate": 0.5},
                "calibrated": {"roc_auc": 0.71, "pr_auc": 0.21, "brier_score": 0.11, "default_rate": 0.5},
            },
            "val_policy": {
                "raw": {"roc_auc": 0.69, "pr_auc": 0.19, "brier_score": 0.13, "default_rate": 0.5},
                "calibrated": {"roc_auc": 0.70, "pr_auc": 0.20, "brier_score": 0.12, "default_rate": 0.5},
            },
            "test": {
                "raw": {"roc_auc": 0.68, "pr_auc": 0.18, "brier_score": 0.14, "default_rate": 0.5},
                "calibrated": {"roc_auc": 0.69, "pr_auc": 0.19, "brier_score": 0.13, "default_rate": 0.5},
            },
        },
    )

    report = asr.run_alt_stacked_reduced_ablation_experiment(
        artifact_dir=str(tmp_path / "artifacts"),
        processed_dir=str(tmp_path / "processed"),
        raw_dir=str(tmp_path / "raw"),
    )
    return report, bundle_payload, artifact_dir, processed_dir, calibration_calls


def test_telco_oof_and_forward_prediction_outputs_match_split_rows(tmp_path, monkeypatch):
    report, bundle_payload, artifact_dir = _run_telco_experiment(tmp_path, monkeypatch)

    train_oof = pd.read_csv(artifact_dir / "telco_submodel_oof_train_predictions.csv")
    val_model = pd.read_csv(artifact_dir / "telco_submodel_val_model_predictions.csv")
    val_policy = pd.read_csv(artifact_dir / "telco_submodel_val_policy_predictions.csv")
    test = pd.read_csv(artifact_dir / "telco_submodel_test_predictions.csv")

    assert len(train_oof) == len(bundle_payload["train_ids"])
    assert train_oof["SK_ID_CURR"].tolist() == bundle_payload["train_ids"]
    assert train_oof["SK_ID_CURR"].nunique() == len(bundle_payload["train_ids"])
    assert train_oof["split_name"].eq("train_oof").all()

    assert val_model["SK_ID_CURR"].tolist() == bundle_payload["val_model_ids"]
    assert val_policy["SK_ID_CURR"].tolist() == bundle_payload["val_policy_ids"]
    assert test["SK_ID_CURR"].tolist() == bundle_payload["test_ids"]
    assert val_model["SK_ID_CURR"].nunique() == len(bundle_payload["val_model_ids"])
    assert val_policy["SK_ID_CURR"].nunique() == len(bundle_payload["val_policy_ids"])
    assert test["SK_ID_CURR"].nunique() == len(bundle_payload["test_ids"])

    required_columns = ["SK_ID_CURR", "TELCO_SUBMODEL_SCORE", "split_name"]
    assert list(train_oof.columns) == required_columns
    assert list(val_model.columns) == required_columns
    assert list(val_policy.columns) == required_columns
    assert list(test.columns) == required_columns

    assert report["sample_counts"]["train"] == len(bundle_payload["train_ids"])
    assert report["sample_counts"]["val_model"] == len(bundle_payload["val_model_ids"])
    assert report["sample_counts"]["val_policy"] == len(bundle_payload["val_policy_ids"])
    assert report["sample_counts"]["test"] == len(bundle_payload["test_ids"])


def test_telco_artifact_and_report_generation_stays_offline_only(tmp_path, monkeypatch):
    report, _, artifact_dir = _run_telco_experiment(tmp_path, monkeypatch)

    expected_files = {
        "telco_submodel.joblib",
        "telco_submodel_oof_train_predictions.csv",
        "telco_submodel_val_model_predictions.csv",
        "telco_submodel_val_policy_predictions.csv",
        "telco_submodel_test_predictions.csv",
        "telco_submodel_report.json",
    }
    assert expected_files.issubset({path.name for path in artifact_dir.iterdir()})
    assert not (artifact_dir / "reduced_model.joblib").exists()
    assert not (artifact_dir / "reduced_calibrator.joblib").exists()

    report_path = artifact_dir / "telco_submodel_report.json"
    assert report_path.exists()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["experiment_name"] == "TELCO_SUBMODEL"
    assert payload["runtime_artifacts_unchanged"] is True
    assert payload["artifacts"]["model"].endswith("telco_submodel.joblib")
    assert report["report_path"] == str(report_path)


def test_telco_fold_logic_uses_train_rows_only(tmp_path, monkeypatch):
    _run_telco_experiment(tmp_path, monkeypatch)

    all_fit_values = {
        int(round(value))
        for batch in _RecordingTelcoModel.fit_feature_values_log
        for value in batch
    }
    allowed_train_values = set(_make_telco_bundle()["train_ids"])
    forbidden_values = set(_make_telco_bundle()["val_model_ids"]) | set(_make_telco_bundle()["val_policy_ids"]) | set(_make_telco_bundle()["test_ids"])

    assert all_fit_values
    assert all_fit_values.issubset(allowed_train_values)
    assert all_fit_values.isdisjoint(forbidden_values)


def test_wallet_oof_and_forward_prediction_outputs_match_split_rows(tmp_path, monkeypatch):
    report, bundle_payload, artifact_dir = _run_wallet_experiment(tmp_path, monkeypatch)

    train_oof = pd.read_csv(artifact_dir / "wallet_submodel_oof_train_predictions.csv")
    val_model = pd.read_csv(artifact_dir / "wallet_submodel_val_model_predictions.csv")
    val_policy = pd.read_csv(artifact_dir / "wallet_submodel_val_policy_predictions.csv")
    test = pd.read_csv(artifact_dir / "wallet_submodel_test_predictions.csv")

    assert len(train_oof) == len(bundle_payload["train_ids"])
    assert train_oof["SK_ID_CURR"].tolist() == bundle_payload["train_ids"]
    assert train_oof["SK_ID_CURR"].nunique() == len(bundle_payload["train_ids"])
    assert train_oof["split_name"].eq("train_oof").all()

    assert val_model["SK_ID_CURR"].tolist() == bundle_payload["val_model_ids"]
    assert val_policy["SK_ID_CURR"].tolist() == bundle_payload["val_policy_ids"]
    assert test["SK_ID_CURR"].tolist() == bundle_payload["test_ids"]
    assert val_model["SK_ID_CURR"].nunique() == len(bundle_payload["val_model_ids"])
    assert val_policy["SK_ID_CURR"].nunique() == len(bundle_payload["val_policy_ids"])
    assert test["SK_ID_CURR"].nunique() == len(bundle_payload["test_ids"])

    required_columns = ["SK_ID_CURR", "WALLET_SUBMODEL_SCORE", "split_name"]
    assert list(train_oof.columns) == required_columns
    assert list(val_model.columns) == required_columns
    assert list(val_policy.columns) == required_columns
    assert list(test.columns) == required_columns

    assert report["sample_counts"]["train"] == len(bundle_payload["train_ids"])
    assert report["sample_counts"]["val_model"] == len(bundle_payload["val_model_ids"])
    assert report["sample_counts"]["val_policy"] == len(bundle_payload["val_policy_ids"])
    assert report["sample_counts"]["test"] == len(bundle_payload["test_ids"])
    assert report["oof_strategy"]["type"] == "blocked_forward_chaining"


def test_wallet_artifact_and_report_generation_stays_offline_only(tmp_path, monkeypatch):
    report, _, artifact_dir = _run_wallet_experiment(tmp_path, monkeypatch)

    expected_files = {
        "wallet_submodel.joblib",
        "wallet_submodel_oof_train_predictions.csv",
        "wallet_submodel_val_model_predictions.csv",
        "wallet_submodel_val_policy_predictions.csv",
        "wallet_submodel_test_predictions.csv",
        "wallet_submodel_report.json",
    }
    assert expected_files.issubset({path.name for path in artifact_dir.iterdir()})
    assert not (artifact_dir / "reduced_model.joblib").exists()
    assert not (artifact_dir / "reduced_calibrator.joblib").exists()

    report_path = artifact_dir / "wallet_submodel_report.json"
    assert report_path.exists()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["experiment_name"] == "WALLET_SUBMODEL"
    assert payload["runtime_artifacts_unchanged"] is True
    assert payload["artifacts"]["model"].endswith("wallet_submodel.joblib")
    assert report["report_path"] == str(report_path)


def test_wallet_fold_logic_uses_train_rows_only_and_reuses_telco_oof_discipline(tmp_path, monkeypatch):
    report, _, _ = _run_wallet_experiment(tmp_path, monkeypatch)

    all_fit_values = {
        int(round(value))
        for batch in _RecordingWalletModel.fit_feature_values_log
        for value in batch
    }
    allowed_train_values = set(_make_telco_bundle()["train_ids"])
    forbidden_values = set(_make_telco_bundle()["val_model_ids"]) | set(_make_telco_bundle()["val_policy_ids"]) | set(_make_telco_bundle()["test_ids"])

    assert all_fit_values
    assert all_fit_values.issubset(allowed_train_values)
    assert report["oof_strategy"]["type"] == "blocked_forward_chaining"
    assert report["oof_strategy"]["uses_train_rows_only"] is True
    assert report["oof_strategy"]["future_rows_never_used_for_earlier_fold_fits"] is True
    assert all_fit_values.isdisjoint(forbidden_values)


def test_ecommerce_oof_and_forward_prediction_outputs_match_split_rows(tmp_path, monkeypatch):
    report, bundle_payload, artifact_dir = _run_ecommerce_experiment(tmp_path, monkeypatch)

    train_oof = pd.read_csv(artifact_dir / "ecommerce_submodel_oof_train_predictions.csv")
    val_model = pd.read_csv(artifact_dir / "ecommerce_submodel_val_model_predictions.csv")
    val_policy = pd.read_csv(artifact_dir / "ecommerce_submodel_val_policy_predictions.csv")
    test = pd.read_csv(artifact_dir / "ecommerce_submodel_test_predictions.csv")

    assert len(train_oof) == len(bundle_payload["train_ids"])
    assert train_oof["SK_ID_CURR"].tolist() == bundle_payload["train_ids"]
    assert train_oof["SK_ID_CURR"].nunique() == len(bundle_payload["train_ids"])
    assert train_oof["split_name"].eq("train_oof").all()

    assert val_model["SK_ID_CURR"].tolist() == bundle_payload["val_model_ids"]
    assert val_policy["SK_ID_CURR"].tolist() == bundle_payload["val_policy_ids"]
    assert test["SK_ID_CURR"].tolist() == bundle_payload["test_ids"]
    assert val_model["SK_ID_CURR"].nunique() == len(bundle_payload["val_model_ids"])
    assert val_policy["SK_ID_CURR"].nunique() == len(bundle_payload["val_policy_ids"])
    assert test["SK_ID_CURR"].nunique() == len(bundle_payload["test_ids"])

    required_columns = ["SK_ID_CURR", "ECOMMERCE_SUBMODEL_SCORE", "split_name"]
    assert list(train_oof.columns) == required_columns
    assert list(val_model.columns) == required_columns
    assert list(val_policy.columns) == required_columns
    assert list(test.columns) == required_columns

    assert report["sample_counts"]["train"] == len(bundle_payload["train_ids"])
    assert report["sample_counts"]["val_model"] == len(bundle_payload["val_model_ids"])
    assert report["sample_counts"]["val_policy"] == len(bundle_payload["val_policy_ids"])
    assert report["sample_counts"]["test"] == len(bundle_payload["test_ids"])
    assert report["oof_strategy"]["type"] == "blocked_forward_chaining"


def test_ecommerce_artifact_and_report_generation_stays_offline_only(tmp_path, monkeypatch):
    report, _, artifact_dir = _run_ecommerce_experiment(tmp_path, monkeypatch)

    expected_files = {
        "ecommerce_submodel.joblib",
        "ecommerce_submodel_oof_train_predictions.csv",
        "ecommerce_submodel_val_model_predictions.csv",
        "ecommerce_submodel_val_policy_predictions.csv",
        "ecommerce_submodel_test_predictions.csv",
        "ecommerce_submodel_report.json",
    }
    assert expected_files.issubset({path.name for path in artifact_dir.iterdir()})
    assert not (artifact_dir / "reduced_model.joblib").exists()
    assert not (artifact_dir / "reduced_calibrator.joblib").exists()

    report_path = artifact_dir / "ecommerce_submodel_report.json"
    assert report_path.exists()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["experiment_name"] == "ECOMMERCE_SUBMODEL"
    assert payload["runtime_artifacts_unchanged"] is True
    assert payload["artifacts"]["model"].endswith("ecommerce_submodel.joblib")
    assert report["report_path"] == str(report_path)


def test_ecommerce_fold_logic_uses_train_rows_only_and_reuses_shared_oof_discipline(tmp_path, monkeypatch):
    report, _, _ = _run_ecommerce_experiment(tmp_path, monkeypatch)

    all_fit_values = {
        int(round(value))
        for batch in _RecordingEcommerceModel.fit_feature_values_log
        for value in batch
    }
    allowed_train_values = set(_make_telco_bundle()["train_ids"])
    forbidden_values = set(_make_telco_bundle()["val_model_ids"]) | set(_make_telco_bundle()["val_policy_ids"]) | set(_make_telco_bundle()["test_ids"])

    assert all_fit_values
    assert all_fit_values.issubset(allowed_train_values)
    assert report["oof_strategy"]["type"] == "blocked_forward_chaining"
    assert report["oof_strategy"]["uses_train_rows_only"] is True
    assert report["oof_strategy"]["future_rows_never_used_for_earlier_fold_fits"] is True
    assert all_fit_values.isdisjoint(forbidden_values)


def test_meta_score_frames_match_split_rows_and_schema(tmp_path, monkeypatch):
    summary, bundle_payload, artifact_dir, _, _ = _run_meta_assembly_experiment(tmp_path, monkeypatch)

    train = pd.read_csv(artifact_dir / "train_meta_scores.csv")
    val_model = pd.read_csv(artifact_dir / "val_model_meta_scores.csv")
    val_policy = pd.read_csv(artifact_dir / "val_policy_meta_scores.csv")
    test = pd.read_csv(artifact_dir / "test_meta_scores.csv")

    required_columns = [
        "SK_ID_CURR",
        "META_SCORE_TELCO",
        "META_SCORE_WALLET",
        "META_SCORE_ECOMMERCE",
        "split_name",
    ]
    assert len(train) == len(bundle_payload["train_ids"])
    assert len(val_model) == len(bundle_payload["val_model_ids"])
    assert len(val_policy) == len(bundle_payload["val_policy_ids"])
    assert len(test) == len(bundle_payload["test_ids"])
    assert list(train.columns) == required_columns
    assert list(val_model.columns) == required_columns
    assert list(val_policy.columns) == required_columns
    assert list(test.columns) == required_columns
    assert train["SK_ID_CURR"].tolist() == bundle_payload["train_ids"]
    assert val_model["SK_ID_CURR"].tolist() == bundle_payload["val_model_ids"]
    assert val_policy["SK_ID_CURR"].tolist() == bundle_payload["val_policy_ids"]
    assert test["SK_ID_CURR"].tolist() == bundle_payload["test_ids"]
    assert train["SK_ID_CURR"].nunique() == len(train)
    assert val_model["SK_ID_CURR"].nunique() == len(val_model)
    assert val_policy["SK_ID_CURR"].nunique() == len(val_policy)
    assert test["SK_ID_CURR"].nunique() == len(test)
    assert summary["split_summary"]["train"]["row_count"] == len(bundle_payload["train_ids"])


def test_meta_assembly_builds_canonical_reduced_before_append_and_stacked_matrices_have_meta_columns(tmp_path, monkeypatch):
    summary, _, _, processed_dir, call_order = _run_meta_assembly_experiment(tmp_path, monkeypatch, track_order=True)

    assert call_order == [
        "build_reduced",
        "build_reduced",
        "build_reduced",
        "build_reduced",
        "append_meta",
        "append_meta",
        "append_meta",
        "append_meta",
    ]

    train_matrix = pd.read_pickle(processed_dir / "train_stacked_reduced.pkl")
    val_model_matrix = pd.read_pickle(processed_dir / "val_model_stacked_reduced.pkl")
    val_policy_matrix = pd.read_pickle(processed_dir / "val_policy_stacked_reduced.pkl")
    test_matrix = pd.read_pickle(processed_dir / "test_stacked_reduced.pkl")

    for matrix in [train_matrix, val_model_matrix, val_policy_matrix, test_matrix]:
        assert "META_SCORE_TELCO" in matrix.columns
        assert "META_SCORE_WALLET" in matrix.columns
        assert "META_SCORE_ECOMMERCE" in matrix.columns

    assert summary["split_summary"]["train"]["canonical_reduced_feature_count"] == 2
    assert summary["split_summary"]["train"]["stacked_reduced_feature_count"] == 5
    assert summary["split_summary"]["train"]["appended_meta_feature_count"] == 3
    assert summary["split_summary"]["train"]["stacked_reduced_feature_count"] == summary["split_summary"]["train"]["canonical_reduced_feature_count"] + 3


def test_stacked_matrix_row_counts_match_canonical_and_runtime_artifacts_remain_untouched(tmp_path, monkeypatch):
    summary, bundle_payload, artifact_dir, processed_dir, _ = _run_meta_assembly_experiment(tmp_path, monkeypatch)

    for split_name, expected_count in [
        ("train", len(bundle_payload["train_ids"])),
        ("val_model", len(bundle_payload["val_model_ids"])),
        ("val_policy", len(bundle_payload["val_policy_ids"])),
        ("test", len(bundle_payload["test_ids"])),
    ]:
        matrix = pd.read_pickle(processed_dir / f"{split_name}_stacked_reduced.pkl")
        assert len(matrix) == expected_count
        assert summary["split_summary"][split_name]["row_count"] == expected_count
        assert summary["split_summary"][split_name]["ids_match_bundle"] is True

    assert not (artifact_dir / "reduced_model.joblib").exists()
    assert not (artifact_dir / "reduced_calibrator.joblib").exists()
    assert (artifact_dir / "meta_score_lineage.json").exists()
    assert (artifact_dir / "stacked_reduced_split_summary.json").exists()


def test_meta_score_lineage_uses_oof_for_train_and_forward_predictions_for_holdouts(tmp_path, monkeypatch):
    _, _, artifact_dir, _, _ = _run_meta_assembly_experiment(tmp_path, monkeypatch)

    lineage = json.loads((artifact_dir / "meta_score_lineage.json").read_text(encoding="utf-8"))

    for side_label in ["telco", "wallet", "ecommerce"]:
        assert lineage["split_sources"]["train"][side_label]["expected_split_name"] == "train_oof"
        assert lineage["split_sources"]["val_model"][side_label]["expected_split_name"] == "val_model"
        assert lineage["split_sources"]["val_policy"][side_label]["expected_split_name"] == "val_policy"
        assert lineage["split_sources"]["test"][side_label]["expected_split_name"] == "test"
        assert "oof_train_predictions" in lineage["split_sources"]["train"][side_label]["path"]
        assert "val_model_predictions" in lineage["split_sources"]["val_model"][side_label]["path"]
        assert "val_policy_predictions" in lineage["split_sources"]["val_policy"][side_label]["path"]
        assert "test_predictions" in lineage["split_sources"]["test"][side_label]["path"]


def test_master_xgb_experiment_generates_predictions_report_and_stays_offline_only(tmp_path, monkeypatch):
    report, _, artifact_dir, _, _ = _run_master_xgb_experiment(tmp_path, monkeypatch)

    expected_files = {
        "alt_stacked_reduced_master_xgb.joblib",
        "alt_stacked_reduced_master_xgb_calibrator.joblib",
        "alt_stacked_reduced_master_xgb_report.json",
        "alt_stacked_reduced_master_xgb_predictions_val_model.csv",
        "alt_stacked_reduced_master_xgb_predictions_val_policy.csv",
        "alt_stacked_reduced_master_xgb_predictions_test.csv",
    }
    assert expected_files.issubset({path.name for path in artifact_dir.iterdir()})
    assert not (artifact_dir / "reduced_model.joblib").exists()
    assert not (artifact_dir / "reduced_calibrator.joblib").exists()

    for name in [
        "alt_stacked_reduced_master_xgb_predictions_val_model.csv",
        "alt_stacked_reduced_master_xgb_predictions_val_policy.csv",
        "alt_stacked_reduced_master_xgb_predictions_test.csv",
    ]:
        frame = pd.read_csv(artifact_dir / name)
        assert list(frame.columns) == [
            "SK_ID_CURR",
            "ALT_STACKED_REDUCED_MASTER_XGB_SCORE_RAW",
            "ALT_STACKED_REDUCED_MASTER_XGB_SCORE_CALIBRATED",
            "split_name",
        ]
        assert frame["ALT_STACKED_REDUCED_MASTER_XGB_SCORE_CALIBRATED"].notna().all()

    report_path = artifact_dir / "alt_stacked_reduced_master_xgb_report.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["experiment_name"] == "ALT_STACKED_REDUCED_MASTER_XGB"
    assert payload["runtime_artifacts_unchanged"] is True
    assert "comparison_vs_frozen_reduced" in payload
    assert report["report_path"] == str(report_path)


def test_master_xgb_uses_meta_score_columns_and_correct_split_roles(tmp_path, monkeypatch):
    report, bundle_payload, _, _, calibration_calls = _run_master_xgb_experiment(tmp_path, monkeypatch)

    assert _RecordingMasterModel.fit_row_counts == [len(bundle_payload["train_ids"])]
    assert _RecordingMasterModel.fit_columns
    fitted_columns = _RecordingMasterModel.fit_columns[0]
    assert "META_SCORE_TELCO" in fitted_columns
    assert "META_SCORE_WALLET" in fitted_columns
    assert "META_SCORE_ECOMMERCE" in fitted_columns

    assert calibration_calls == [
        {
            "y_len": len(bundle_payload["val_policy_ids"]),
            "raw_len": len(bundle_payload["val_policy_ids"]),
        }
    ]
    assert _RecordingMasterCalibrator.predict_call_row_counts == [
        len(bundle_payload["val_policy_ids"]),
        len(bundle_payload["val_model_ids"]),
        len(bundle_payload["test_ids"]),
    ]
    assert _RecordingMasterModel.predict_row_counts == [
        len(bundle_payload["val_model_ids"]),
        len(bundle_payload["val_policy_ids"]),
        len(bundle_payload["test_ids"]),
    ]
    assert report["feature_summary"]["contains_meta_columns"]["META_SCORE_TELCO"] is True
    assert report["feature_summary"]["contains_meta_columns"]["META_SCORE_WALLET"] is True
    assert report["feature_summary"]["contains_meta_columns"]["META_SCORE_ECOMMERCE"] is True


def test_master_xgb_report_includes_comparison_section_against_frozen_reduced(tmp_path, monkeypatch):
    _, _, artifact_dir, _, _ = _run_master_xgb_experiment(tmp_path, monkeypatch)

    payload = json.loads((artifact_dir / "alt_stacked_reduced_master_xgb_report.json").read_text(encoding="utf-8"))
    comparison = payload["comparison_vs_frozen_reduced"]

    assert "frozen_reduced_val_model_roc_auc" in comparison
    assert "frozen_reduced_test_roc_auc" in comparison
    assert "frozen_reduced_brier" in comparison
    assert "delta_vs_frozen_reduced" in comparison
    assert "val_model_roc_auc" in comparison["delta_vs_frozen_reduced"]
    assert "test_roc_auc" in comparison["delta_vs_frozen_reduced"]
    assert "test_pr_auc" in comparison["delta_vs_frozen_reduced"]
    assert "calibrated_test_brier" in comparison["delta_vs_frozen_reduced"]


def test_ablation_configurations_include_exact_intended_meta_columns(tmp_path, monkeypatch):
    report, _, _, _, calibration_calls = _run_ablation_experiment(tmp_path, monkeypatch)

    expected = {
        "BASE_REDUCED_ONLY": [],
        "REDUCED_PLUS_TELCO": ["META_SCORE_TELCO"],
        "REDUCED_PLUS_WALLET": ["META_SCORE_WALLET"],
        "REDUCED_PLUS_ECOMMERCE": ["META_SCORE_ECOMMERCE"],
        "REDUCED_PLUS_TELCO_WALLET": ["META_SCORE_TELCO", "META_SCORE_WALLET"],
        "REDUCED_PLUS_ALL_THREE": ["META_SCORE_TELCO", "META_SCORE_WALLET", "META_SCORE_ECOMMERCE"],
    }

    for config_name, expected_meta_columns in expected.items():
        payload = report["configurations"][config_name]
        assert payload["feature_summary"]["selected_meta_columns"] == expected_meta_columns
        assert [col for col in payload["feature_summary"]["contains_meta_columns"] if payload["feature_summary"]["contains_meta_columns"][col]] == expected_meta_columns

    assert len(calibration_calls) == len(expected)


def test_base_reduced_only_has_no_meta_columns_and_row_counts_match(tmp_path, monkeypatch):
    report, bundle_payload, _, _, _ = _run_ablation_experiment(tmp_path, monkeypatch)

    base_payload = report["configurations"]["BASE_REDUCED_ONLY"]

    assert base_payload["feature_summary"]["selected_meta_columns"] == []
    assert base_payload["feature_summary"]["contains_meta_columns"] == {
        "META_SCORE_TELCO": False,
        "META_SCORE_WALLET": False,
        "META_SCORE_ECOMMERCE": False,
    }
    assert base_payload["sample_counts"] == {
        "train": len(bundle_payload["train_ids"]),
        "val_model": len(bundle_payload["val_model_ids"]),
        "val_policy": len(bundle_payload["val_policy_ids"]),
        "test": len(bundle_payload["test_ids"]),
    }
    assert all(base_payload["row_count_checks"].values())


def test_ablation_report_generation_stays_offline_only(tmp_path, monkeypatch):
    report, _, artifact_dir, _, _ = _run_ablation_experiment(tmp_path, monkeypatch)

    assert (artifact_dir / "alt_stacked_reduced_ablation_report.json").exists()
    assert (artifact_dir / "alt_stacked_reduced_ablation_summary.md").exists()
    assert not (artifact_dir / "reduced_model.joblib").exists()
    assert not (artifact_dir / "reduced_calibrator.joblib").exists()

    payload = json.loads((artifact_dir / "alt_stacked_reduced_ablation_report.json").read_text(encoding="utf-8"))
    assert payload["experiment_name"] == "ALT_STACKED_REDUCED_ABLATION"
    assert payload["runtime_artifacts_unchanged"] is True
    assert "recommendation" in payload
    assert report["report_path"] == str(artifact_dir / "alt_stacked_reduced_ablation_report.json")
