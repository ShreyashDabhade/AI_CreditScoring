import os
from pathlib import Path

import numpy as np
import pandas as pd


def _make_application_df(sk_ids: list[int], include_target: bool = True) -> pd.DataFrame:
    from src.feature_engineering import CATEGORICAL_MODEL_COLS, FAIRNESS_ONLY_COLS, NUMERIC_RAW_COLS

    n = len(sk_ids)
    df = pd.DataFrame({"SK_ID_CURR": sk_ids})

    numeric_defaults = {
        "AMT_INCOME_TOTAL_CAPPED": 120000.0,
        "AMT_CREDIT": 250000.0,
        "AMT_ANNUITY": 25000.0,
        "AMT_GOODS_PRICE": 220000.0,
        "DAYS_BIRTH": -12000.0,
        "DAYS_EMPLOYED": -1500.0,
        "DAYS_REGISTRATION": -3000.0,
        "DAYS_ID_PUBLISH": -2000.0,
        "DAYS_LAST_PHONE_CHANGE": -1000.0,
        "REGION_POPULATION_RELATIVE": 0.02,
        "EXT_SOURCE_1": 0.2,
        "EXT_SOURCE_2": 0.4,
        "EXT_SOURCE_3": 0.6,
        "CNT_FAM_MEMBERS": 2.0,
        "OWN_CAR_AGE": 5.0,
        "OBS_30_CNT_SOCIAL_CIRCLE": 1.0,
        "DEF_30_CNT_SOCIAL_CIRCLE": 0.0,
        "OBS_60_CNT_SOCIAL_CIRCLE": 1.0,
        "DEF_60_CNT_SOCIAL_CIRCLE": 0.0,
        "AMT_REQ_CREDIT_BUREAU_HOUR": 0.0,
        "AMT_REQ_CREDIT_BUREAU_DAY": 0.0,
        "AMT_REQ_CREDIT_BUREAU_WEEK": 1.0,
        "AMT_REQ_CREDIT_BUREAU_MON": 1.0,
        "AMT_REQ_CREDIT_BUREAU_QRT": 0.0,
        "AMT_REQ_CREDIT_BUREAU_YEAR": 1.0,
    }
    for col in NUMERIC_RAW_COLS:
        df[col] = np.full(n, numeric_defaults[col], dtype=float)

    categorical_defaults = {
        "NAME_CONTRACT_TYPE": "Cash loans",
        "NAME_TYPE_SUITE": "Unaccompanied",
        "NAME_EDUCATION_TYPE": "Higher education",
        "NAME_FAMILY_STATUS": "Married",
        "OCCUPATION_TYPE": "Laborers",
        "ORGANIZATION_TYPE": "Business Entity Type 3",
        "WEEKDAY_APPR_PROCESS_START": "MONDAY",
    }
    for col in CATEGORICAL_MODEL_COLS:
        df[col] = [categorical_defaults[col]] * n

    fairness_defaults = {
        "REGION_RATING_CLIENT_W_CITY": 2.0,
        "NAME_INCOME_TYPE": "Working",
        "NAME_HOUSING_TYPE": "House / apartment",
        "FLAG_OWN_CAR": "N",
        "FLAG_OWN_REALTY": "Y",
        "CNT_CHILDREN": 1.0,
    }
    for col in FAIRNESS_ONLY_COLS:
        df[col] = [fairness_defaults[col]] * n

    df["AMT_INCOME_TOTAL"] = np.full(n, 130000.0, dtype=float)
    df["DAYS_EMPLOYED_ANOM"] = np.zeros(n, dtype="int8")
    df["CODE_GENDER"] = ["M"] * n
    if include_target:
        df["TARGET"] = np.array([0, 1, 0, 1][:n], dtype=int)
    return df


def _write_raw_tables(raw_dir: Path, sk_ids: list[int]) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    prev_rows = []
    bureau_rows = []
    inst_rows = []
    pos_rows = []
    cc_rows = []

    for idx, sk_id in enumerate(sk_ids, start=1):
        sk_prev = sk_id * 10
        prev_rows.append(
            {
                "SK_ID_PREV": sk_prev,
                "SK_ID_CURR": sk_id,
                "NAME_CONTRACT_STATUS": "Approved" if idx % 2 else "Refused",
                "AMT_APPLICATION": 240000.0 + idx,
                "AMT_CREDIT": 230000.0 + idx,
                "AMT_GOODS_PRICE": 220000.0 + idx,
                "DAYS_DECISION": -100.0 - idx,
                "RATE_DOWN_PAYMENT": 0.1,
            }
        )
        bureau_rows.append(
            {
                "SK_ID_CURR": sk_id,
                "CREDIT_ACTIVE": "Active" if idx % 2 else "Closed",
                "AMT_CREDIT_SUM": 50000.0 + idx,
                "AMT_CREDIT_SUM_DEBT": 10000.0 + idx,
                "AMT_CREDIT_SUM_OVERDUE": 0.0,
                "CREDIT_DAY_OVERDUE": 0.0,
                "DAYS_CREDIT": -200.0 - idx,
                "CNT_CREDIT_PROLONG": 0.0,
            }
        )
        inst_rows.append(
            {
                "SK_ID_PREV": sk_prev,
                "DAYS_ENTRY_PAYMENT": -10.0,
                "DAYS_INSTALMENT": -12.0,
                "AMT_INSTALMENT": 1000.0,
                "AMT_PAYMENT": 1000.0,
            }
        )
        pos_rows.append(
            {
                "SK_ID_PREV": sk_prev,
                "SK_DPD": 0.0,
                "SK_DPD_DEF": 0.0,
                "NAME_CONTRACT_STATUS": "Completed" if idx % 2 else "Active",
                "CNT_INSTALMENT_FUTURE": 2.0,
            }
        )
        cc_rows.append(
            {
                "SK_ID_PREV": sk_prev,
                "AMT_CREDIT_LIMIT_ACTUAL": 50000.0,
                "AMT_BALANCE": 12000.0,
                "AMT_INST_MIN_REGULARITY": 500.0,
                "AMT_PAYMENT_TOTAL_CURRENT": 550.0,
                "SK_DPD": 0.0,
                "AMT_DRAWINGS_ATM_CURRENT": 100.0,
                "AMT_DRAWINGS_CURRENT": 250.0,
            }
        )

    pd.DataFrame(bureau_rows).to_csv(raw_dir / "bureau.csv", index=False)
    pd.DataFrame(prev_rows).to_csv(raw_dir / "previous_application.csv", index=False)
    pd.DataFrame(inst_rows).to_csv(raw_dir / "installments_payments.csv", index=False)
    pd.DataFrame(pos_rows).to_csv(raw_dir / "POS_CASH_balance.csv", index=False)
    pd.DataFrame(cc_rows).to_csv(raw_dir / "credit_card_balance.csv", index=False)


def test_m2_exports_support_m4_contract():
    from src.feature_engineering import FrozenFeatureBuilder, build_full, fit_full_builder, pool_rare_categories

    assert FrozenFeatureBuilder is not None
    assert callable(build_full)
    assert callable(fit_full_builder)
    assert callable(pool_rare_categories)


def test_fit_full_builder_is_pure_without_save_path(tmp_path):
    from src.feature_engineering import fit_full_builder

    raw_dir = tmp_path / "raw"
    sk_ids = [300001, 300002, 300003, 300004]
    _write_raw_tables(raw_dir, sk_ids)
    train_df = _make_application_df(sk_ids)

    project_root = Path(__file__).resolve().parents[1]
    shared_artifacts = project_root / "artifacts"
    before = sorted(path.name for path in shared_artifacts.iterdir()) if shared_artifacts.exists() else []
    builder = fit_full_builder(train_df, raw_dir=str(raw_dir), save_path=None)
    after = sorted(path.name for path in shared_artifacts.iterdir()) if shared_artifacts.exists() else []

    assert builder.tier == "FULL"
    assert before == after


def test_m4_derives_pooled_groups_via_m2_contract(tmp_path, monkeypatch):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mplconfig"))
    from src.fairness_audit import derive_fairness_groups

    df = pd.DataFrame(
        {
            "AMT_INCOME_TOTAL": [50000.0, 70000.0, 90000.0],
            "REGION_RATING_CLIENT_W_CITY": [1, 2, 3],
            "NAME_INCOME_TYPE": ["Working", "Pensioner", "Student"],
            "NAME_HOUSING_TYPE": ["House / apartment", "Rented apartment", "With parents"],
            "FLAG_OWN_CAR": ["Y", "N", "N"],
            "FLAG_OWN_REALTY": ["Y", "Y", "N"],
            "CNT_CHILDREN": [0.0, 1.0, 3.0],
        }
    )
    result = derive_fairness_groups(df, 60000.0, 80000.0)

    assert "INCOME_TYPE_POOLED" in result.columns
    assert "HOUSING_TYPE_POOLED" in result.columns
    assert set(result["INCOME_TYPE_POOLED"]) == {"OTHER"}
    assert set(result["HOUSING_TYPE_POOLED"]) == {"OTHER"}


def test_m2_full_builder_and_m4_audit_flow(tmp_path, monkeypatch):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mplconfig"))
    from src.feature_engineering import FAIRNESS_ONLY_COLS, fit_full_builder, build_full
    from src.fairness_audit import run_full_fairness_audit

    raw_dir = tmp_path / "raw"
    sk_ids = [100001, 100002, 100003, 100004]
    _write_raw_tables(raw_dir, sk_ids)
    train_df = _make_application_df(sk_ids)
    test_df = _make_application_df(sk_ids)

    builder = fit_full_builder(train_df, raw_dir=str(raw_dir))
    X_full = build_full(test_df, builder, raw_dir=str(raw_dir))

    assert X_full.shape[0] == len(test_df)
    assert all(np.issubdtype(dtype, np.number) for dtype in X_full.dtypes)
    assert all(col not in X_full.columns for col in FAIRNESS_ONLY_COLS)

    rng = np.random.default_rng(7)
    audit_df = test_df[
        [
            "TARGET",
            "AMT_INCOME_TOTAL",
            "REGION_RATING_CLIENT_W_CITY",
            "NAME_INCOME_TYPE",
            "NAME_HOUSING_TYPE",
            "FLAG_OWN_CAR",
            "FLAG_OWN_REALTY",
            "CNT_CHILDREN",
        ]
    ].copy()
    audit_df["CALIBRATED_PD"] = rng.uniform(0.0, 1.0, len(audit_df))
    audit_df["DECISION"] = np.where(
        audit_df["CALIBRATED_PD"] < 0.15,
        "APPROVE",
        np.where(audit_df["CALIBRATED_PD"] < 0.35, "REVIEW", "DECLINE"),
    )

    out_dir = tmp_path / "fairness_outputs"
    passed = run_full_fairness_audit(
        audit_df,
        float(train_df["AMT_INCOME_TOTAL"].quantile(1 / 3)),
        float(train_df["AMT_INCOME_TOTAL"].quantile(2 / 3)),
        str(out_dir),
    )

    assert isinstance(passed, bool)
    for name in [
        "audit_primary.csv",
        "audit_secondary.csv",
        "audit_tertiary.csv",
        "fairness_summary_card.png",
    ]:
        assert (out_dir / name).exists(), name


def test_builder_shim_loads_or_fits_m2_builder_for_m4(tmp_path, monkeypatch):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mplconfig"))
    from src.fairness_audit import _load_or_fit_full_builder
    from src.feature_engineering import FrozenFeatureBuilder

    raw_dir = tmp_path / "raw"
    builder_path = tmp_path / "full_feature_builder.joblib"
    sk_ids = [200001, 200002, 200003, 200004]
    _write_raw_tables(raw_dir, sk_ids)
    train_df = _make_application_df(sk_ids)

    built = _load_or_fit_full_builder(
        train_df,
        raw_dir=str(raw_dir),
        builder_path=str(builder_path),
    )
    loaded = _load_or_fit_full_builder(
        train_df,
        raw_dir=str(raw_dir),
        builder_path=str(builder_path),
    )

    assert isinstance(built, FrozenFeatureBuilder)
    assert isinstance(loaded, FrozenFeatureBuilder)
    assert builder_path.exists()
    assert builder_path.with_suffix(".manifest.json").exists()


def test_builder_artifacts_strict_loader_validates_saved_builders(tmp_path):
    import json

    from src.builder_artifacts import load_validated_builders
    from src.feature_engineering import fit_full_builder, fit_reduced_builder

    artifact_dir = tmp_path / "artifacts"
    processed_dir = tmp_path / "processed"
    raw_dir = tmp_path / "raw"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = processed_dir / "processed_artifact_manifest.json"
    manifest_path.write_text(
        json.dumps({"processed_manifest_fingerprint": "fp-test-001"}),
        encoding="utf-8",
    )

    sk_ids = [400001, 400002, 400003, 400004]
    _write_raw_tables(raw_dir, sk_ids)
    train_df = _make_application_df(sk_ids)

    fit_full_builder(
        train_df,
        raw_dir=str(raw_dir),
        save_path=str(artifact_dir / "full_feature_builder.joblib"),
        processed_manifest_path=str(manifest_path),
    )
    fit_reduced_builder(
        train_df,
        save_path=str(artifact_dir / "reduced_feature_builder.joblib"),
        processed_manifest_path=str(manifest_path),
    )

    builders = load_validated_builders(
        artifact_dir=str(artifact_dir),
        processed_dir=str(processed_dir),
        strict_artifacts=True,
    )

    assert builders["FULL"].tier == "FULL"
    assert builders["REDUCED"].tier == "REDUCED"
