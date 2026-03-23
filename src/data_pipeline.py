"""Module 1 — Data Pipeline.

This module handles:
- Data loading
- Schema enforcement
- Data trap handling (Stage 2)
- Proxy recency sort & ordered split (Stage 3)
"""

from datetime import datetime, timezone
import os
from typing import Any
import uuid

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import json

from src.builder_artifacts import (
    current_git_commit,
    dataframe_schema_hash,
    dataset_fingerprint,
    processed_manifest_fingerprint,
    processed_manifest_path,
    quarantine_existing_paths,
)


# ================================
# CONSTANTS
# ================================

LOCKED_SCORING_TABLES: set[str] = {
    "application_train.csv",
    "bureau.csv",
    "previous_application.csv",
    "installments_payments.csv",
    "POS_CASH_balance.csv",
    "credit_card_balance.csv"
}

ADVERSARIAL_ONLY_TABLES: set[str] = {
    "application_test.csv"
}

TRAIN_SCHEMA: dict[str, str] = {
    "SK_ID_CURR": "int64",
    "CNT_CHILDREN": "int64",
    "CNT_FAM_MEMBERS": "float64",
    "AMT_INCOME_TOTAL": "float64",
    "AMT_CREDIT": "float64",
    "AMT_ANNUITY": "float64",
    "AMT_GOODS_PRICE": "float64",
    "DAYS_BIRTH": "float64",
    "DAYS_EMPLOYED": "float64",
    "DAYS_REGISTRATION": "float64",
    "DAYS_ID_PUBLISH": "float64",
    "DAYS_LAST_PHONE_CHANGE": "float64",
    "EXT_SOURCE_1": "float64",
    "EXT_SOURCE_2": "float64",
    "EXT_SOURCE_3": "float64",
    "NAME_CONTRACT_TYPE": "object",
    "NAME_TYPE_SUITE": "object",
    "NAME_EDUCATION_TYPE": "object",
    "NAME_FAMILY_STATUS": "object",
    "OCCUPATION_TYPE": "object",
    "ORGANIZATION_TYPE": "object",
    "WEEKDAY_APPR_PROCESS_START": "object",
    "NAME_INCOME_TYPE": "object",
    "NAME_HOUSING_TYPE": "object",
    "FLAG_OWN_CAR": "object",
    "FLAG_OWN_REALTY": "object",
    "REGION_RATING_CLIENT_W_CITY": "float64"
}

PREV_SENTINEL_DAY_COLS: list[str] = [
    "DAYS_FIRST_DRAWING",
    "DAYS_FIRST_DUE",
    "DAYS_LAST_DUE_1ST_VERSION",
    "DAYS_LAST_DUE",
    "DAYS_TERMINATION"
]

# All DAYS_* and MONTHS_BALANCE fields are relative offsets,
# not calendar timestamps.
RELATIVE_TIME_COLS: set[str] = {
    "DAYS_BIRTH", "DAYS_EMPLOYED", "DAYS_REGISTRATION",
    "DAYS_ID_PUBLISH", "DAYS_DECISION", "MONTHS_BALANCE",
    "DAYS_INSTALMENT", "DAYS_ENTRY_PAYMENT"
}

# ================================
# FUNCTIONS
# ================================

def enforce_locked_tables(loaded_names: list[str]) -> None:
    unexpected = set(loaded_names) - LOCKED_SCORING_TABLES - ADVERSARIAL_ONLY_TABLES
    if unexpected:
        raise ValueError(f"Unexpected table(s) referenced: {sorted(unexpected)}")


def enforce_schema(df: pd.DataFrame, schema: dict[str, str]) -> pd.DataFrame:
    df_copy = df.copy()
    for col, dtype in schema.items():
        if col in df_copy.columns:
            df_copy[col] = df_copy[col].astype(np.dtype(dtype))
    return df_copy


def _absolute_median(series: pd.Series) -> float:
    median = series.abs().median(skipna=True)
    if pd.isna(median):
        return 0.0
    return float(median)


def _build_staleness_keys(app_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    tmp = app_df.copy()
    days_id_publish_fill = _absolute_median(tmp["DAYS_ID_PUBLISH"])
    days_registration_fill = _absolute_median(tmp["DAYS_REGISTRATION"])
    tmp["__STALENESS_1"] = tmp["DAYS_ID_PUBLISH"].abs().fillna(days_id_publish_fill)
    tmp["__STALENESS_2"] = tmp["DAYS_REGISTRATION"].abs().fillna(days_registration_fill)
    return tmp, {
        "days_id_publish_fill": days_id_publish_fill,
        "days_registration_fill": days_registration_fill,
    }


def _split_summary_entry(df: pd.DataFrame) -> dict[str, float | int]:
    return {
        "rows": int(len(df)),
        "positive_count": int(df["TARGET"].sum()),
        "target_rate": float(df["TARGET"].mean()),
        "mean_abs_days_id_publish": float(df["DAYS_ID_PUBLISH"].abs().mean()),
        "mean_abs_days_registration": float(df["DAYS_REGISTRATION"].abs().mean()),
        "days_employed_anom_rate": float(df["DAYS_EMPLOYED_ANOM"].mean()),
    }


def _split_schema_signature(df: pd.DataFrame) -> list[tuple[str, str]]:
    return [(column, str(df.dtypes[column])) for column in df.columns]


def _validate_row_preservation(splits: dict[str, pd.DataFrame], expected_rows: int) -> None:
    actual_rows = sum(len(df) for df in splits.values())
    assert actual_rows == expected_rows, (
        "Split row preservation failed: "
        f"expected {expected_rows}, found {actual_rows}"
    )


def _validate_split_key_integrity(splits: dict[str, pd.DataFrame]) -> None:
    seen_ids: set[int] = set()
    for split_name, df in splits.items():
        dupes = int(df["SK_ID_CURR"].duplicated().sum())
        assert dupes == 0, f"{split_name} contains {dupes} duplicate SK_ID_CURR values"
        split_ids = set(df["SK_ID_CURR"].tolist())
        overlap = seen_ids.intersection(split_ids)
        assert not overlap, f"{split_name} overlaps prior splits on SK_ID_CURR"
        seen_ids.update(split_ids)


def _validate_schema_consistency(splits: dict[str, pd.DataFrame]) -> None:
    split_items = list(splits.items())
    reference_name, reference_df = split_items[0]
    reference_signature = _split_schema_signature(reference_df)
    for split_name, df in split_items[1:]:
        assert _split_schema_signature(df) == reference_signature, (
            f"{split_name} schema differs from {reference_name}"
        )


def _validate_monotonic_staleness(split_summary: dict[str, dict[str, float | int]]) -> None:
    ordered_split_names = ["train", "val_model", "val_policy", "test"]
    publish_means = [
        float(split_summary[split_name]["mean_abs_days_id_publish"])
        for split_name in ordered_split_names
    ]
    registration_means = [
        float(split_summary[split_name]["mean_abs_days_registration"])
        for split_name in ordered_split_names
    ]
    assert all(
        left >= right for left, right in zip(publish_means, publish_means[1:])
    ), "DAYS_ID_PUBLISH staleness is not monotonic oldest-to-newest across splits"
    assert all(
        left >= right for left, right in zip(registration_means, registration_means[1:])
    ), "DAYS_REGISTRATION staleness is not monotonic oldest-to-newest across splits"


def _validate_processing_contract(
    splits: dict[str, pd.DataFrame],
    income_cap: float,
) -> None:
    train_income_cap = float(splits["train"]["AMT_INCOME_TOTAL"].quantile(0.99))
    assert np.isclose(train_income_cap, income_cap), (
        f"income_cap mismatch: expected {train_income_cap}, found {income_cap}"
    )

    for split_name, df in splits.items():
        sentinel_count = int((df["DAYS_EMPLOYED"] == 365243).fillna(False).sum())
        assert sentinel_count == 0, f"{split_name} still contains DAYS_EMPLOYED sentinel values"
        expected_capped = np.minimum(df["AMT_INCOME_TOTAL"], income_cap)
        mismatches = int(
            (~np.isclose(df["AMT_INCOME_TOTAL_CAPPED"], expected_capped, equal_nan=True)).sum()
        )
        assert mismatches == 0, f"{split_name} has {mismatches} AMT_INCOME_TOTAL_CAPPED mismatches"
        assert int(df["DAYS_EMPLOYED_ANOM"].sum()) == int(df["DAYS_EMPLOYED"].isna().sum()), (
            f"{split_name} DAYS_EMPLOYED anomaly flag does not match null count"
        )


def _validate_adversarial_contract(
    adv_train_df: pd.DataFrame,
    adv_val_df: pd.DataFrame,
    income_cap: float,
) -> dict[str, Any]:
    frames = {"adv_train": adv_train_df, "adv_val": adv_val_df}
    reference_non_label_columns = [
        column for column in adv_train_df.columns if column != "ADV_LABEL"
    ]

    for name, df in frames.items():
        assert "ADV_LABEL" in df.columns, f"{name} is missing ADV_LABEL"
        assert "TARGET" not in df.columns, f"{name} must not contain TARGET"
        assert "AMT_INCOME_TOTAL_CAPPED" in df.columns, (
            f"{name} is missing AMT_INCOME_TOTAL_CAPPED"
        )
        assert "DAYS_EMPLOYED_ANOM" in df.columns, f"{name} is missing DAYS_EMPLOYED_ANOM"
        assert int(df["SK_ID_CURR"].duplicated().sum()) == 0, (
            f"{name} contains duplicate SK_ID_CURR values"
        )
        assert (
            [column for column in df.columns if column != "ADV_LABEL"]
            == reference_non_label_columns
        ), f"{name} column order differs from adv_train"
        expected_capped = np.minimum(df["AMT_INCOME_TOTAL"], income_cap)
        mismatches = int(
            (~np.isclose(df["AMT_INCOME_TOTAL_CAPPED"], expected_capped, equal_nan=True)).sum()
        )
        assert mismatches == 0, f"{name} has {mismatches} adversarial cap mismatches"

    return {
        "adv_train_rows": int(len(adv_train_df)),
        "adv_val_rows": int(len(adv_val_df)),
        "income_cap_used": float(income_cap),
        "schema_hash": dataframe_schema_hash(adv_train_df),
        "diagnostic_only": True,
        "git_commit": current_git_commit(),
    }


def _build_processed_manifest(
    splits: dict[str, pd.DataFrame],
    adversarial_summary: dict[str, Any],
    *,
    income_cap: float,
    application_train_cleaned_rows: int,
) -> dict[str, Any]:
    manifest_id = f"processed-{uuid.uuid4().hex[:12]}"
    manifest: dict[str, Any] = {
        "manifest_version": 2,
        "processed_manifest_id": manifest_id,
        "lineage": manifest_id,
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "git_commit": current_git_commit(),
        "application_train_cleaned_rows": int(application_train_cleaned_rows),
        "income_cap": float(income_cap),
        "income_cap_p99": float(income_cap),
        "split_summary": {
            split_name: _split_summary_entry(df)
            for split_name, df in splits.items()
        },
        "split_fingerprints": {
            split_name: dataset_fingerprint(df)
            for split_name, df in splits.items()
        },
        "split_schema_hashes": {
            split_name: dataframe_schema_hash(df)
            for split_name, df in splits.items()
        },
        "adversarial_summary": adversarial_summary,
    }
    manifest["processed_manifest_fingerprint"] = processed_manifest_fingerprint(manifest)
    return manifest


def _build_data_quality_report(
    split_summary: dict[str, dict[str, float | int]],
    income_cap: float,
) -> dict[str, Any]:
    return {
        "split_summary": split_summary,
        "train_fit_artifacts": {
            "income_cap_p99": float(income_cap),
        },
        "diagnostic_notes": [
            "Split summaries reflect the corrected forward proxy-time regime.",
        ],
    }


# ================================
# TRAP D — MISSING POLICY FUNCTIONS
# ================================

def fit_missing_policy(train_df: pd.DataFrame):
    miss_rate = train_df.isna().mean()
    flag_cols = miss_rate[miss_rate >= 0.05].index.tolist()

    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = train_df.select_dtypes(exclude=[np.number]).columns.tolist()

    num_medians = train_df[
        [c for c in numeric_cols if c in flag_cols]
    ].median(numeric_only=True).to_dict()

    cat_modes = {
        c: (
            train_df[c].mode(dropna=True).iloc[0]
            if train_df[c].dropna().shape[0] > 0
            else "MISSING"
        )
        for c in categorical_cols if c in flag_cols
    }

    return (flag_cols, num_medians, cat_modes)


def apply_missing_policy(
    df: pd.DataFrame,
    flag_cols: list[str],
    num_medians: dict[str, float],
    cat_modes: dict[str, str],
    for_linear_model: bool = False
) -> pd.DataFrame:

    df = df.copy()

    for col in flag_cols:
        if col in df.columns:
            df[f"{col}_IS_MISSING"] = df[col].isna().astype("int8")

    if for_linear_model:
        for col, val in num_medians.items():
            if col in df.columns:
                df[col] = df[col].fillna(val)

        for col, val in cat_modes.items():
            if col in df.columns:
                df[col] = df[col].fillna(val)

    return df


# ================================
# STAGE 3 FUNCTIONS
# ================================

# ordered holdout using Home Credit proxy staleness;
# train is oldest, test is newest
def proxy_recency_sort(app_df: pd.DataFrame) -> pd.DataFrame:
    tmp, _ = _build_staleness_keys(app_df)
    tmp = tmp.sort_values(
        ["__STALENESS_1", "__STALENESS_2", "SK_ID_CURR"],
        ascending=[False, False, True]
    )
    tmp = tmp.drop(columns=["__STALENESS_1", "__STALENESS_2"])
    return tmp.reset_index(drop=True)


def ordered_split_60_10_10_20(df: pd.DataFrame):
    n = len(df)

    i60 = int(n * 0.60)
    i70 = int(n * 0.70)
    i80 = int(n * 0.80)

    train = df.iloc[:i60].copy()
    val_model = df.iloc[i60:i70].copy()
    val_policy = df.iloc[i70:i80].copy()
    test = df.iloc[i80:].copy()

    return (train, val_model, val_policy, test)


def build_adversarial_dataset(train_app: pd.DataFrame, test_app: pd.DataFrame):
    train_aligned = train_app.drop(columns=["TARGET"], errors="ignore").copy()
    test_aligned = test_app.drop(columns=["TARGET"], errors="ignore").copy()
    shared_columns = [
        column
        for column in train_aligned.columns
        if column in test_aligned.columns
    ]
    train_aligned = train_aligned[shared_columns]
    test_aligned = test_aligned[shared_columns]

    # Step 1: Balance dataset
    min_size = min(len(train_app), len(test_app))

    train_sample = train_aligned.sample(n=min_size, random_state=42)
    test_sample = test_aligned.sample(n=min_size, random_state=42)

    # Step 2: Assign labels
    adv_train = train_sample.copy()
    adv_train["ADV_LABEL"] = 0

    adv_test = test_sample.copy()
    adv_test["ADV_LABEL"] = 1

    # Step 3: Combine
    combined = pd.concat([adv_train, adv_test], ignore_index=True)

    # Step 4: Split
    adv_X_train, adv_X_val = train_test_split(
        combined,
        test_size=0.20,
        stratify=combined["ADV_LABEL"],
        random_state=42
    )

    adv_X_train = adv_X_train.reset_index(drop=True)
    adv_X_val = adv_X_val.reset_index(drop=True)
    return (adv_X_train, adv_X_val)


# ================================
# MAIN EXECUTION BLOCK
# ================================

if __name__ == "__main__":

    DATA_RAW_DIR = os.environ.get("DATA_RAW_DIR", "data/raw/")

    filenames = [
        "application_train.csv",
        "bureau.csv",
        "previous_application.csv",
        "installments_payments.csv",
        "POS_CASH_balance.csv",
        "credit_card_balance.csv",
        "application_test.csv"
    ]

    # Load datasets
    app_train = pd.read_csv(os.path.join(DATA_RAW_DIR, filenames[0]))
    bureau = pd.read_csv(os.path.join(DATA_RAW_DIR, filenames[1]))
    prev_app = pd.read_csv(os.path.join(DATA_RAW_DIR, filenames[2]))
    inst = pd.read_csv(os.path.join(DATA_RAW_DIR, filenames[3]))
    pos_cash = pd.read_csv(os.path.join(DATA_RAW_DIR, filenames[4]))
    cc = pd.read_csv(os.path.join(DATA_RAW_DIR, filenames[5]))
    app_test = pd.read_csv(os.path.join(DATA_RAW_DIR, filenames[6]))

    # Enforce table whitelist
    enforce_locked_tables(filenames)

    # Apply schema
    app_train = enforce_schema(app_train, TRAIN_SCHEMA)
    app_test = enforce_schema(app_test, TRAIN_SCHEMA)

    # ================================
    # TRAP A — DAYS_EMPLOYED FIX
    # ================================
    for app_df in [app_train, app_test]:
        app_df["DAYS_EMPLOYED_ANOM"] = (
            app_df["DAYS_EMPLOYED"] == 365243
        ).astype("int8")
        app_df["DAYS_EMPLOYED"] = app_df["DAYS_EMPLOYED"].replace(365243, np.nan)

    # ================================
    # TRAP B — PREV_APP FIX
    # ================================
    for col in PREV_SENTINEL_DAY_COLS:
        if col in prev_app.columns:
            prev_app[f"{col}_ANOM"] = (prev_app[col] == 365243).astype("int8")
            prev_app[col] = prev_app[col].replace(365243, np.nan)

    # ================================
    # TRAP E — SCHEMA ASSERTION
    # ================================
    assert app_train["SK_ID_CURR"].dtype == np.int64, \
        "SK_ID_CURR dtype enforcement failed"

    # ================================
    # STAGE 3 — SORT + SPLIT
    # ================================
    app_sorted = proxy_recency_sort(app_train)

    train, val_model, val_policy, test = ordered_split_60_10_10_20(app_sorted)
    scoring_splits = {
        "train": train,
        "val_model": val_model,
        "val_policy": val_policy,
        "test": test,
    }

    # ================================
    # TRAP C — INCOME CAP
    # ================================
    income_cap = train["AMT_INCOME_TOTAL"].quantile(0.99)

    for partition in [app_train, app_test, train, val_model, val_policy, test]:
        partition["AMT_INCOME_TOTAL_CAPPED"] = partition["AMT_INCOME_TOTAL"].clip(
            upper=income_cap
        )

    # ================================
    # ADVERSARIAL DATASET
    # ================================
    adv_train_df, adv_val_df = build_adversarial_dataset(app_train, app_test)

    # ================================
    # PRINT SHAPES
    # ================================
    print(f"{filenames[0]}: {app_train.shape[0]} rows, {app_train.shape[1]} cols")
    print(f"{filenames[1]}: {bureau.shape[0]} rows, {bureau.shape[1]} cols")
    print(f"{filenames[2]}: {prev_app.shape[0]} rows, {prev_app.shape[1]} cols")
    print(f"{filenames[3]}: {inst.shape[0]} rows, {inst.shape[1]} cols")
    print(f"{filenames[4]}: {pos_cash.shape[0]} rows, {pos_cash.shape[1]} cols")
    print(f"{filenames[5]}: {cc.shape[0]} rows, {cc.shape[1]} cols")
    print(f"{filenames[6]}: {app_test.shape[0]} rows, {app_test.shape[1]} cols")

    print("\nSplit Sizes:")
    print(f"train:      {len(train)} rows")
    print(f"val_model:  {len(val_model)} rows")
    print(f"val_policy: {len(val_policy)} rows")
    print(f"test:       {len(test)} rows")
    print(f"adv_train:  {len(adv_train_df)} rows")
    print(f"adv_val:    {len(adv_val_df)} rows")

    # ================================
    # VALIDATION CHECKS (STAGE 3)
    # ================================
    _validate_row_preservation(scoring_splits, len(app_train))

    ratio = len(train) / len(app_train)
    assert 0.599 <= ratio <= 0.601

    _validate_split_key_integrity(scoring_splits)
    _validate_schema_consistency(scoring_splits)
    _validate_processing_contract(scoring_splits, float(income_cap))

    split_summary = {
        split_name: _split_summary_entry(df)
        for split_name, df in scoring_splits.items()
    }
    _validate_monotonic_staleness(split_summary)
    print("\nForward proxy-time split summary (oldest -> newest):")
    print(pd.DataFrame(split_summary).T.to_string())

    adversarial_summary = _validate_adversarial_contract(
        adv_train_df,
        adv_val_df,
        float(income_cap),
    )
    
    # ================================
    # STAGE 4 — DIRECTORY SETUP
    # ================================

    DATA_PROCESSED_DIR = os.environ.get(
        "DATA_PROCESSED_DIR", "data/processed/"
    )
    data_quality_report_path = os.path.join("data", "data_quality_report.json")
    processed_manifest_output_path = processed_manifest_path(DATA_PROCESSED_DIR)

    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)

    # ================================
    # SERIALIZATION FUNCTION
    # ================================

    def serialize_dataframe(df: pd.DataFrame, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump(df, f, protocol=4)
        print(f"Saved {path}: {df.shape[0]} rows, {df.shape[1]} cols")

    # ================================
    # SAVE DATAFRAMES
    # ================================
    processed_output_paths = [
        os.path.join(DATA_PROCESSED_DIR, "train.pkl"),
        os.path.join(DATA_PROCESSED_DIR, "val_model.pkl"),
        os.path.join(DATA_PROCESSED_DIR, "val_policy.pkl"),
        os.path.join(DATA_PROCESSED_DIR, "test.pkl"),
        os.path.join(DATA_PROCESSED_DIR, "app_test_adv.pkl"),
        os.path.join(DATA_PROCESSED_DIR, "income_cap.joblib"),
        processed_manifest_output_path,
        data_quality_report_path,
    ]
    quarantine_existing_paths(
        processed_output_paths,
        os.path.join("artifacts", "quarantine", "module1_outputs"),
    )

    serialize_dataframe(train, os.path.join(DATA_PROCESSED_DIR, "train.pkl"))
    serialize_dataframe(val_model, os.path.join(DATA_PROCESSED_DIR, "val_model.pkl"))
    serialize_dataframe(val_policy, os.path.join(DATA_PROCESSED_DIR, "val_policy.pkl"))
    serialize_dataframe(test, os.path.join(DATA_PROCESSED_DIR, "test.pkl"))

    adv_path = os.path.join(DATA_PROCESSED_DIR, "app_test_adv.pkl")
    with open(adv_path, "wb") as f:
        pickle.dump(
            {"adv_train": adv_train_df, "adv_val": adv_val_df},
            f,
            protocol=4
        )
    print(f"Saved {adv_path}")

    # ================================
    # SAVE INCOME CAP
    # ================================

    joblib.dump(income_cap, os.path.join(DATA_PROCESSED_DIR, "income_cap.joblib"))
    print(f"income_cap = {income_cap:.2f} saved")

    processed_manifest = _build_processed_manifest(
        scoring_splits,
        adversarial_summary,
        income_cap=float(income_cap),
        application_train_cleaned_rows=len(app_train),
    )
    with open(processed_manifest_output_path, "w", encoding="utf-8") as f:
        json.dump(processed_manifest, f, indent=2, sort_keys=True)
    print(f"Saved {processed_manifest_output_path}")

    # ================================
    # RELOAD VERIFICATION
    # ================================

    def verify_dataframe(path: str, original_df: pd.DataFrame):
        with open(path, "rb") as f:
            reloaded = pickle.load(f)

        assert isinstance(reloaded, pd.DataFrame), \
            f"{path} did not reload as DataFrame"

        assert reloaded.shape == original_df.shape, \
            f"{path} shape mismatch after reload"


    verify_dataframe(os.path.join(DATA_PROCESSED_DIR, "train.pkl"), train)
    verify_dataframe(os.path.join(DATA_PROCESSED_DIR, "val_model.pkl"), val_model)
    verify_dataframe(os.path.join(DATA_PROCESSED_DIR, "val_policy.pkl"), val_policy)
    verify_dataframe(os.path.join(DATA_PROCESSED_DIR, "test.pkl"), test)

    with open(adv_path, "rb") as f:
        adv_loaded = pickle.load(f)

    assert isinstance(adv_loaded, dict)
    assert set(adv_loaded.keys()) == {"adv_train", "adv_val"}

    loaded_income_cap = joblib.load(os.path.join(DATA_PROCESSED_DIR, "income_cap.joblib"))
    assert isinstance(loaded_income_cap, float)
    assert np.isclose(loaded_income_cap, processed_manifest["income_cap"])

    with open(processed_manifest_output_path, "r", encoding="utf-8") as f:
        saved_manifest = json.load(f)
    assert (
        saved_manifest["processed_manifest_fingerprint"]
        == processed_manifest_fingerprint(saved_manifest)
    ), "Saved processed manifest fingerprint mismatch"

    print("\nAll serialization checks passed.")
    
    # ================================
    # STAGE 5 — EDA DIRECTORY
    # ================================

    EDA_PLOTS_DIR = os.environ.get(
        "EDA_PLOTS_DIR", "notebooks/eda_plots/"
    )

    os.makedirs(EDA_PLOTS_DIR, exist_ok=True)
    eda_plot_paths = [
        os.path.join(EDA_PLOTS_DIR, "target_distribution.png"),
        os.path.join(EDA_PLOTS_DIR, "missing_value_heatmap.png"),
        os.path.join(EDA_PLOTS_DIR, "ext_source_correlation.png"),
        os.path.join(EDA_PLOTS_DIR, "days_employed_anomaly.png"),
        os.path.join(EDA_PLOTS_DIR, "income_outliers.png"),
    ]
    quarantine_existing_paths(
        eda_plot_paths,
        os.path.join("artifacts", "quarantine", "module1_eda"),
    )
    
    # Plot 1 — Target Distribution
    
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = train["TARGET"].value_counts().sort_index()

    ax.bar(
    ["No default (0)", "Default (1)"],
    counts.values,
    color=["#1D9E75", "#D85A30"]
    )

    ax.set_title("Target distribution (train partition)")
    ax.set_ylabel("Count")

    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v:,} ({v/len(train)*100:.1f}%)",
            fontsize=10, ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(EDA_PLOTS_DIR + "target_distribution.png", dpi=150)
    plt.close()
    
    # Plot 2 — Missing Heatmap
    
    miss_rate = app_train.isna().mean().sort_values(ascending=False)
    miss_top = miss_rate[miss_rate > 0].head(40)

    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        miss_top.to_frame().T,
        annot=False,
        cmap="YlOrRd",
        ax=ax,
        vmin=0,
        vmax=1,
        cbar_kws={"label": "Missing rate"}
    )

    ax.set_title("Missing value rates — application_train (top 40)")
    ax.set_xlabel("Column")
    plt.xticks(rotation=90, fontsize=7)

    plt.tight_layout()
    plt.savefig(EDA_PLOTS_DIR + "missing_value_heatmap.png", dpi=150)
    plt.close()
    
    #Plot 3 — Correlation
    cols = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "TARGET"]
    sub = train[cols].dropna()
    corr = sub.corr()

    fig, ax = plt.subplots(figsize=(5, 4))

    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        ax=ax,
        vmin=-1,
        vmax=1
    )

    ax.set_title("EXT_SOURCE & TARGET correlations (train)")

    plt.tight_layout()
    plt.savefig(EDA_PLOTS_DIR + "ext_source_correlation.png", dpi=150)
    plt.close()
    
    #Plot 4 — DAYS_EMPLOYED anomaly
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    raw_vals = app_train["DAYS_EMPLOYED"].copy()
    raw_vals_with_sentinel = raw_vals.copy()

    raw_vals_with_sentinel[
        app_train["DAYS_EMPLOYED_ANOM"] == 1
    ] = 365243

    axes[0].hist(
        raw_vals_with_sentinel.dropna(),
        bins=50,
        color="#378ADD",
        edgecolor="none"
    )

    axes[0].set_title("DAYS_EMPLOYED — with sentinel (365243)")

    axes[1].hist(
        app_train["DAYS_EMPLOYED"].dropna(),
        bins=50,
        color="#1D9E75",
        edgecolor="none"
    )

    axes[1].set_title("DAYS_EMPLOYED — after Trap A fix")

    for ax in axes:
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")

    plt.suptitle("Trap A: DAYS_EMPLOYED sentinel removal")

    plt.tight_layout()
    plt.savefig(EDA_PLOTS_DIR + "days_employed_anomaly.png", dpi=150)
    plt.close()
    
    #Plot 5 — Income Outliers
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(
        train["AMT_INCOME_TOTAL"].clip(upper=2e6).dropna(),
        bins=60,
        color="#7F77DD",
        edgecolor="none"
    )

    axes[0].set_title("AMT_INCOME_TOTAL — raw (clipped at 2M)")

    axes[1].hist(
        train["AMT_INCOME_TOTAL_CAPPED"].dropna(),
        bins=60,
        color="#1D9E75",
        edgecolor="none"
    )

    axes[1].axvline(
        x=income_cap,
        color="#D85A30",
        linestyle="--",
        linewidth=1.5,
        label=f"99th pct = {income_cap:,.0f}"
    )

    axes[1].legend(fontsize=9)
    axes[1].set_title("AMT_INCOME_TOTAL — after Trap C cap")

    for ax in axes:
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")

    plt.suptitle("Trap C: Income outlier capping")

    plt.tight_layout()
    plt.savefig(EDA_PLOTS_DIR + "income_outliers.png", dpi=150)
    plt.close()
    
    # Data Quality Report
    
    report = _build_data_quality_report(split_summary, float(income_cap))

    with open(data_quality_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("data_quality_report.json saved")
