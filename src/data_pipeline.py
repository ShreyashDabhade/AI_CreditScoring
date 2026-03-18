"""Module 1 — Data Pipeline.

Loads 7 raw Home Credit CSVs, enforces an allowlist, applies 6 data traps
(sentinel removal, schema enforcement, income capping), sorts by proxy
recency, splits 60/10/10/20, prepares adversarial-validation data,
serialises five DataFrames + income_cap to disk, and generates EDA plots
and a data-quality report.
"""

from __future__ import annotations

import json
import os
import pickle
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────

LOCKED_SCORING_TABLES: set[str] = {
    "application_train.csv",
    "bureau.csv",
    "previous_application.csv",
    "installments_payments.csv",
    "POS_CASH_balance.csv",
    "credit_card_balance.csv",
}

ADVERSARIAL_ONLY_TABLES: set[str] = {"application_test.csv"}

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
    "REGION_RATING_CLIENT_W_CITY": "float64",
}

PREV_SENTINEL_DAY_COLS: list[str] = [
    "DAYS_FIRST_DRAWING",
    "DAYS_FIRST_DUE",
    "DAYS_LAST_DUE_1ST_VERSION",
    "DAYS_LAST_DUE",
    "DAYS_TERMINATION",
]

# All DAYS_* and MONTHS_BALANCE fields are relative offsets,
# not calendar timestamps.
RELATIVE_TIME_COLS: set[str] = {
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "DAYS_REGISTRATION",
    "DAYS_ID_PUBLISH",
    "DAYS_DECISION",
    "MONTHS_BALANCE",
    "DAYS_INSTALMENT",
    "DAYS_ENTRY_PAYMENT",
}


# ──────────────────────────────────────────────────────────────────────
# Stage 1 — Table Loading & Enforcement
# ──────────────────────────────────────────────────────────────────────

def enforce_locked_tables(loaded_names: list[str]) -> None:
    """Raise ``ValueError`` if any table name is outside the allowlist."""
    unexpected = set(loaded_names) - LOCKED_SCORING_TABLES - ADVERSARIAL_ONLY_TABLES
    if unexpected:
        raise ValueError(f"Unexpected table(s) referenced: {sorted(unexpected)}")


def enforce_schema(df: pd.DataFrame, schema: dict[str, str]) -> pd.DataFrame:
    """Cast columns of *df* to the dtypes specified in *schema*."""
    df = df.copy()
    for col, dt in schema.items():
        if col in df.columns:
            df[col] = df[col].astype(dt)
    return df


# ──────────────────────────────────────────────────────────────────────
# Stage 2 — Data Trap Handling
# ──────────────────────────────────────────────────────────────────────

def apply_trap_a(app_train: pd.DataFrame) -> pd.DataFrame:
    """Trap A — Replace DAYS_EMPLOYED == 365243 with NaN; create flag."""
    app_train = app_train.copy()
    app_train["DAYS_EMPLOYED_ANOM"] = (
        app_train["DAYS_EMPLOYED"] == 365243
    ).astype("int8")
    app_train["DAYS_EMPLOYED"] = app_train["DAYS_EMPLOYED"].replace(
        365243, np.nan
    )
    return app_train


def apply_trap_b(prev_app: pd.DataFrame) -> pd.DataFrame:
    """Trap B — Replace sentinels in prev_app day columns."""
    prev_app = prev_app.copy()
    for col in PREV_SENTINEL_DAY_COLS:
        if col in prev_app.columns:
            prev_app[f"{col}_ANOM"] = (prev_app[col] == 365243).astype("int8")
            prev_app[col] = prev_app[col].replace(365243, np.nan)
    return prev_app


# Trap C applied post-split in Stage 3


def fit_missing_policy(
    train_df: pd.DataFrame,
) -> tuple[list[str], dict[str, float], dict[str, str]]:
    """Trap D part 1 — Identify columns with high missingness on training data."""
    miss_rate = train_df.isna().mean()
    flag_cols = miss_rate[miss_rate >= 0.05].index.tolist()

    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = train_df.select_dtypes(exclude=[np.number]).columns.tolist()

    num_medians = (
        train_df[[c for c in numeric_cols if c in flag_cols]]
        .median(numeric_only=True)
        .to_dict()
    )
    cat_modes: dict[str, str] = {}
    for c in categorical_cols:
        if c in flag_cols:
            non_null = train_df[c].dropna()
            if len(non_null) > 0:
                cat_modes[c] = str(train_df[c].mode(dropna=True).iloc[0])
            else:
                cat_modes[c] = "MISSING"

    return (flag_cols, num_medians, cat_modes)


def apply_missing_policy(
    df: pd.DataFrame,
    flag_cols: list[str],
    num_medians: dict[str, float],
    cat_modes: dict[str, str],
    for_linear_model: bool = False,
) -> pd.DataFrame:
    """Trap D part 2 — Add _IS_MISSING flags; optionally fill NaN."""
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


# ──────────────────────────────────────────────────────────────────────
# Stage 3 — Proxy-Recency Sort & Ordered Split
# ──────────────────────────────────────────────────────────────────────

# ordered holdout using Home Credit recency proxies;
# not true calendar-time validation
def proxy_recency_sort(app_df: pd.DataFrame) -> pd.DataFrame:
    """Sort by abs(DAYS_ID_PUBLISH) ASC, abs(DAYS_REGISTRATION) ASC, SK_ID_CURR ASC."""
    tmp = app_df.copy()
    tmp["__RECENCY_1"] = tmp["DAYS_ID_PUBLISH"].abs().fillna(
        tmp["DAYS_ID_PUBLISH"].abs().median()
    )
    tmp["__RECENCY_2"] = tmp["DAYS_REGISTRATION"].abs().fillna(
        tmp["DAYS_REGISTRATION"].abs().median()
    )
    tmp = tmp.sort_values(
        ["__RECENCY_1", "__RECENCY_2", "SK_ID_CURR"],
        ascending=[True, True, True],
    )
    tmp = tmp.drop(columns=["__RECENCY_1", "__RECENCY_2"])
    return tmp.reset_index(drop=True)


def ordered_split_60_10_10_20(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Deterministic ordered split at 60/10/10/20."""
    n = len(df)
    i60 = int(n * 0.60)
    i70 = int(n * 0.70)
    i80 = int(n * 0.80)
    train = df.iloc[:i60].copy()
    val_model = df.iloc[i60:i70].copy()
    val_policy = df.iloc[i70:i80].copy()
    test = df.iloc[i80:].copy()
    return (train, val_model, val_policy, test)


def build_adversarial_dataset(
    train_app: pd.DataFrame,
    test_app: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Label train rows=0, test rows=1; stratified 80/20 split."""
    adv_train = train_app.copy()
    adv_train["ADV_LABEL"] = 0
    adv_test = test_app.copy()
    adv_test["ADV_LABEL"] = 1
    combined = pd.concat([adv_train, adv_test], ignore_index=True)
    adv_X_train, adv_X_val = train_test_split(
        combined,
        test_size=0.20,
        stratify=combined["ADV_LABEL"],
        random_state=42,
    )
    return (
        adv_X_train.reset_index(drop=True),
        adv_X_val.reset_index(drop=True),
    )


# ──────────────────────────────────────────────────────────────────────
# Stage 4 — Serialization
# ──────────────────────────────────────────────────────────────────────

def serialize_dataframe(df: pd.DataFrame, path: str) -> None:
    """Pickle a DataFrame with protocol=4."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(df, f, protocol=4)
    print(f"Saved {path}: {df.shape[0]} rows, {df.shape[1]} cols")


# ──────────────────────────────────────────────────────────────────────
# Stage 5 — EDA Plots & Data Quality Report
# ──────────────────────────────────────────────────────────────────────

def generate_eda_plots(
    app_train: pd.DataFrame,
    train: pd.DataFrame,
    income_cap: float,
    eda_dir: str,
) -> None:
    """Generate and save all 5 EDA PNG plots."""
    os.makedirs(eda_dir, exist_ok=True)

    # Plot 1 — target_distribution.png
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = train["TARGET"].value_counts().sort_index()
    ax.bar(
        ["No default (0)", "Default (1)"],
        counts.values,
        color=["#1D9E75", "#D85A30"],
    )
    ax.set_title("Target distribution (train partition)")
    ax.set_ylabel("Count")
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v:,} ({v / len(train) * 100:.1f}%)",
                fontsize=10, ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(os.path.join(eda_dir, "target_distribution.png"), dpi=150)
    plt.close()

    # Plot 2 — missing_value_heatmap.png
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
        cbar_kws={"label": "Missing rate"},
    )
    ax.set_title("Missing value rates — application_train (top 40)")
    ax.set_xlabel("Column")
    plt.xticks(rotation=90, fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(eda_dir, "missing_value_heatmap.png"), dpi=150)
    plt.close()

    # Plot 3 — ext_source_correlation.png
    cols = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "TARGET"]
    sub = train[cols].dropna()
    corr = sub.corr()
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax, vmin=-1, vmax=1)
    ax.set_title("EXT_SOURCE & TARGET correlations (train)")
    plt.tight_layout()
    plt.savefig(os.path.join(eda_dir, "ext_source_correlation.png"), dpi=150)
    plt.close()

    # Plot 4 — days_employed_anomaly.png
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    raw_vals_with_sentinel = app_train["DAYS_EMPLOYED"].copy()
    raw_vals_with_sentinel[app_train["DAYS_EMPLOYED_ANOM"] == 1] = 365243
    axes[0].hist(raw_vals_with_sentinel.dropna(), bins=50, color="#378ADD", edgecolor="none")
    axes[0].set_title("DAYS_EMPLOYED — with sentinel (365243)")
    axes[1].hist(app_train["DAYS_EMPLOYED"].dropna(), bins=50, color="#1D9E75", edgecolor="none")
    axes[1].set_title("DAYS_EMPLOYED — after Trap A fix")
    for ax in axes:
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")
    plt.suptitle("Trap A: DAYS_EMPLOYED sentinel removal")
    plt.tight_layout()
    plt.savefig(os.path.join(eda_dir, "days_employed_anomaly.png"), dpi=150)
    plt.close()

    # Plot 5 — income_outliers.png
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(
        train["AMT_INCOME_TOTAL"].clip(upper=2e6).dropna(),
        bins=60,
        color="#7F77DD",
        edgecolor="none",
    )
    axes[0].set_title("AMT_INCOME_TOTAL — raw (clipped at 2M)")
    axes[1].hist(
        train["AMT_INCOME_TOTAL_CAPPED"].dropna(),
        bins=60,
        color="#1D9E75",
        edgecolor="none",
    )
    axes[1].axvline(
        x=income_cap,
        color="#D85A30",
        linestyle="--",
        linewidth=1.5,
        label=f"99th pct = {income_cap:,.0f}",
    )
    axes[1].legend(fontsize=9)
    axes[1].set_title("AMT_INCOME_TOTAL — after Trap C cap")
    for ax in axes:
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")
    plt.suptitle("Trap C: Income outlier capping")
    plt.tight_layout()
    plt.savefig(os.path.join(eda_dir, "income_outliers.png"), dpi=150)
    plt.close()

    print(f"All 5 EDA plots saved to {eda_dir}")


def generate_data_quality_report(
    app_train: pd.DataFrame,
    train: pd.DataFrame,
    income_cap: float,
    split_sizes: dict[str, int],
    output_path: str,
) -> None:
    """Generate and save data_quality_report.json."""
    report = {
        "dataset": "application_train",
        "total_rows": int(len(app_train)),
        "total_cols": int(len(app_train.columns)),
        "target_default_rate": float(round(train["TARGET"].mean(), 4)),
        "class_counts": {
            str(k): int(v) for k, v in train["TARGET"].value_counts().items()
        },
        "sentinel_counts": {
            "DAYS_EMPLOYED_365243": int(app_train["DAYS_EMPLOYED_ANOM"].sum())
        },
        "missing_rates": {
            col: float(round(rate, 4))
            for col, rate in app_train.isna().mean().items()
            if rate > 0
        },
        "income_cap_p99": float(round(income_cap, 2)),
        "split_sizes": split_sizes,
    }
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"data_quality_report.json saved to {output_path}")


# ──────────────────────────────────────────────────────────────────────
# Main execution
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    DATA_RAW_DIR = os.environ.get("DATA_RAW_DIR", "data/raw/")
    DATA_PROCESSED_DIR = os.environ.get("DATA_PROCESSED_DIR", "data/processed/")
    EDA_PLOTS_DIR = os.environ.get("EDA_PLOTS_DIR", "notebooks/eda_plots/")

    # ── Stage 1 — Load & enforce ─────────────────────────────────────
    print("=" * 60)
    print("Stage 1 — Table Loading & Enforcement")
    print("=" * 60)

    filenames = [
        "application_train.csv",
        "bureau.csv",
        "previous_application.csv",
        "installments_payments.csv",
        "POS_CASH_balance.csv",
        "credit_card_balance.csv",
        "application_test.csv",
    ]
    enforce_locked_tables(filenames)

    app_train = pd.read_csv(os.path.join(DATA_RAW_DIR, "application_train.csv"))
    bureau = pd.read_csv(os.path.join(DATA_RAW_DIR, "bureau.csv"))
    prev_app = pd.read_csv(os.path.join(DATA_RAW_DIR, "previous_application.csv"))
    inst = pd.read_csv(os.path.join(DATA_RAW_DIR, "installments_payments.csv"))
    pos_cash = pd.read_csv(os.path.join(DATA_RAW_DIR, "POS_CASH_balance.csv"))
    cc = pd.read_csv(os.path.join(DATA_RAW_DIR, "credit_card_balance.csv"))
    app_test = pd.read_csv(os.path.join(DATA_RAW_DIR, "application_test.csv"))

    for name, df in [
        ("application_train.csv", app_train),
        ("bureau.csv", bureau),
        ("previous_application.csv", prev_app),
        ("installments_payments.csv", inst),
        ("POS_CASH_balance.csv", pos_cash),
        ("credit_card_balance.csv", cc),
        ("application_test.csv", app_test),
    ]:
        print(f"  {name}: {df.shape[0]} rows, {df.shape[1]} cols")

    app_train = enforce_schema(app_train, TRAIN_SCHEMA)
    assert app_train["SK_ID_CURR"].dtype == np.int64, "SK_ID_CURR dtype enforcement failed"
    print("  Schema enforcement applied to application_train")
    print("Stage 1 DONE\n")

    # ── Stage 2 — Data Trap Handling ─────────────────────────────────
    print("=" * 60)
    print("Stage 2 — Data Trap Handling")
    print("=" * 60)

    app_train = apply_trap_a(app_train)
    assert app_train["DAYS_EMPLOYED"].eq(365243).sum() == 0
    assert app_train["DAYS_EMPLOYED_ANOM"].dtype == np.int8
    print(f"  Trap A: {app_train['DAYS_EMPLOYED_ANOM'].sum()} sentinels fixed")

    prev_app = apply_trap_b(prev_app)
    for col in PREV_SENTINEL_DAY_COLS:
        if col in prev_app.columns:
            assert prev_app[col].eq(365243).sum() == 0
            assert prev_app[f"{col}_ANOM"].dtype == np.int8
    print("  Trap B: prev_app sentinel columns fixed")

    # Trap E — already applied via enforce_schema in Stage 1
    assert app_train["SK_ID_CURR"].dtype == np.int64, "Trap E failed"
    print("  Trap E: Schema assertion OK")
    print("Stage 2 DONE\n")

    # ── Stage 3 — Sort, Split, Trap C, Adversarial ───────────────────
    print("=" * 60)
    print("Stage 3 — Proxy-Recency Sort & Ordered Split")
    print("=" * 60)

    app_sorted = proxy_recency_sort(app_train)
    train, val_model, val_policy, test = ordered_split_60_10_10_20(app_sorted)

    # Trap C — income cap (fit on train only)
    income_cap = float(train["AMT_INCOME_TOTAL"].quantile(0.99))
    for part in [train, val_model, val_policy, test]:
        part["AMT_INCOME_TOTAL_CAPPED"] = part["AMT_INCOME_TOTAL"].clip(
            upper=income_cap
        )

    total = len(train) + len(val_model) + len(val_policy) + len(test)
    assert total == len(app_train), "Split totals mismatch"
    print(f"  income_cap (99th pct, train): {income_cap:,.2f}")
    print(f"  train:      {len(train)} rows")
    print(f"  val_model:  {len(val_model)} rows")
    print(f"  val_policy: {len(val_policy)} rows")
    print(f"  test:       {len(test)} rows")

    # Adversarial validation dataset
    adv_train_df, adv_val_df = build_adversarial_dataset(app_train, app_test)
    print(f"  adv_train:  {len(adv_train_df)} rows")
    print(f"  adv_val:    {len(adv_val_df)} rows")
    print("Stage 3 DONE\n")

    # ── Stage 4 — Serialization ──────────────────────────────────────
    print("=" * 60)
    print("Stage 4 — Serialization")
    print("=" * 60)

    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)

    serialize_dataframe(train, os.path.join(DATA_PROCESSED_DIR, "train.pkl"))
    serialize_dataframe(val_model, os.path.join(DATA_PROCESSED_DIR, "val_model.pkl"))
    serialize_dataframe(val_policy, os.path.join(DATA_PROCESSED_DIR, "val_policy.pkl"))
    serialize_dataframe(test, os.path.join(DATA_PROCESSED_DIR, "test.pkl"))

    # Adversarial as dict
    adv_path = os.path.join(DATA_PROCESSED_DIR, "app_test_adv.pkl")
    with open(adv_path, "wb") as f:
        pickle.dump({"adv_train": adv_train_df, "adv_val": adv_val_df}, f, protocol=4)
    print(f"Saved {adv_path}: dict with adv_train ({len(adv_train_df)} rows) "
          f"and adv_val ({len(adv_val_df)} rows)")

    # income_cap scalar
    joblib.dump(income_cap, os.path.join(DATA_PROCESSED_DIR, "income_cap.joblib"))
    print(f"income_cap = {income_cap:.2f} saved")

    # Reload verification
    for name, expected_shape in [
        ("train.pkl", train.shape),
        ("val_model.pkl", val_model.shape),
        ("val_policy.pkl", val_policy.shape),
        ("test.pkl", test.shape),
    ]:
        path = os.path.join(DATA_PROCESSED_DIR, name)
        reloaded = pickle.load(open(path, "rb"))
        assert isinstance(reloaded, pd.DataFrame), f"{path} not a DataFrame"
        assert reloaded.shape == expected_shape, f"{path} shape mismatch"
    adv_reloaded = pickle.load(open(adv_path, "rb"))
    assert isinstance(adv_reloaded, dict)
    assert set(adv_reloaded.keys()) == {"adv_train", "adv_val"}
    cap_reloaded = joblib.load(os.path.join(DATA_PROCESSED_DIR, "income_cap.joblib"))
    assert isinstance(cap_reloaded, float)
    print("  Reload verification PASSED")
    print("Stage 4 DONE\n")

    # ── Stage 5 — EDA Plots & Data Quality Report ────────────────────
    print("=" * 60)
    print("Stage 5 — EDA Plots & Data Quality Report")
    print("=" * 60)

    generate_eda_plots(app_train, train, income_cap, EDA_PLOTS_DIR)

    split_sizes = {
        "train": int(len(train)),
        "val_model": int(len(val_model)),
        "val_policy": int(len(val_policy)),
        "test": int(len(test)),
    }
    generate_data_quality_report(
        app_train, train, income_cap, split_sizes,
        "data/data_quality_report.json",
    )

    # Verify plots
    for png_name in [
        "target_distribution.png",
        "missing_value_heatmap.png",
        "ext_source_correlation.png",
        "days_employed_anomaly.png",
        "income_outliers.png",
    ]:
        path = os.path.join(EDA_PLOTS_DIR, png_name)
        assert os.path.exists(path), f"Missing: {path}"
        assert os.path.getsize(path) > 10_000, f"Too small: {path}"
    print("  All 5 EDA PNGs verified (exist and >10KB)")

    # Verify JSON
    with open("data/data_quality_report.json") as f:
        rpt = json.load(f)
    assert sum(rpt["split_sizes"].values()) == rpt["total_rows"]
    print("  data_quality_report.json verified")
    print("Stage 5 DONE\n")
    print("=" * 60)
    print("MODULE 1 COMPLETE — all outputs verified")
    print("=" * 60)
