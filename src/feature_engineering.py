"""MasterMind Module 2 - Feature Engineering."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.runtime_verification import (
    column_sequence_hash,
    dataframe_fingerprint,
    rare_map_schema_hash,
    safe_git_commit,
)

CATEGORICAL_MODEL_COLS = [
    "NAME_CONTRACT_TYPE",
    "NAME_TYPE_SUITE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "OCCUPATION_TYPE",
    "ORGANIZATION_TYPE",
    "WEEKDAY_APPR_PROCESS_START",
]
NUMERIC_RAW_COLS = [
    "AMT_INCOME_TOTAL_CAPPED",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "DAYS_REGISTRATION",
    "DAYS_ID_PUBLISH",
    "DAYS_LAST_PHONE_CHANGE",
    "REGION_POPULATION_RELATIVE",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "CNT_FAM_MEMBERS",
    "OWN_CAR_AGE",
    "OBS_30_CNT_SOCIAL_CIRCLE",
    "DEF_30_CNT_SOCIAL_CIRCLE",
    "OBS_60_CNT_SOCIAL_CIRCLE",
    "DEF_60_CNT_SOCIAL_CIRCLE",
    "AMT_REQ_CREDIT_BUREAU_HOUR",
    "AMT_REQ_CREDIT_BUREAU_DAY",
    "AMT_REQ_CREDIT_BUREAU_WEEK",
    "AMT_REQ_CREDIT_BUREAU_MON",
    "AMT_REQ_CREDIT_BUREAU_QRT",
    "AMT_REQ_CREDIT_BUREAU_YEAR",
]
FAIRNESS_ONLY_COLS = [
    "REGION_RATING_CLIENT_W_CITY",
    "NAME_INCOME_TYPE",
    "NAME_HOUSING_TYPE",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "CNT_CHILDREN",
]
FORBIDDEN_COLS = ["SK_ID_CURR", "TARGET", "CODE_GENDER", "SK_ID_PREV"]
APPLICATION_REQUIRED_INPUT_COLS = NUMERIC_RAW_COLS + CATEGORICAL_MODEL_COLS + [
    "DAYS_EMPLOYED_ANOM"
]
ENGINEERED_APP_FEATURE_COLS = [
    "AGE_YEARS",
    "CREDIT_INCOME_RATIO",
    "ANNUITY_INCOME_RATIO",
    "GOODS_CREDIT_RATIO",
    "CREDIT_TERM_RATIO",
    "EMPLOYED_BIRTH_RATIO",
    "ID_PUBLISH_REG_RATIO",
    "EXT_SOURCE_MEAN",
    "EXT_SOURCE_STD",
    "SOCIAL_CIRCLE_SUM",
    "BUREAU_REQUEST_SUM",
    "DAYS_EMPLOYED_ANOM",
]
BUREAU_AGG_COLS = [
    "BUREAU_LOAN_COUNT",
    "BUREAU_ACTIVE_COUNT",
    "BUREAU_CLOSED_COUNT",
    "BUREAU_AMT_CREDIT_SUM_SUM",
    "BUREAU_AMT_CREDIT_SUM_DEBT_SUM",
    "BUREAU_DEBT_TO_CREDIT_RATIO",
    "BUREAU_AMT_CREDIT_SUM_OVERDUE_SUM",
    "BUREAU_CREDIT_DAY_OVERDUE_MAX",
    "BUREAU_DAYS_CREDIT_MAX",
    "BUREAU_CNT_CREDIT_PROLONG_SUM",
    "BUREAU_ACTIVE_SHARE",
    "BUREAU_AMT_CREDIT_MAX_OVERDUE_MAX",
    "BUREAU_ACTIVE_DEBT_RATIO",
    "BUREAU_DAYS_CREDIT_MEAN",
    "BUREAU_DAYS_CREDIT_MIN",
    "BB_MONTHS_COUNT_MEAN",
    "BB_MAX_STATUS_MAX",
    "BB_MAX_STATUS_MEAN",
    "BB_ADVERSE_ACCOUNT_RATE",
    "BB_ADVERSE_RATE_MEAN",
]
PREVIOUS_AGG_COLS = [
    "PREV_APP_COUNT",
    "PREV_APPROVED_COUNT",
    "PREV_REFUSED_COUNT",
    "PREV_APPROVAL_RATE",
    "PREV_REFUSAL_RATE",
    "PREV_AMT_APPLICATION_MEAN",
    "PREV_AMT_CREDIT_MEAN",
    "PREV_AMT_GOODS_PRICE_MEAN",
    "PREV_APP_CREDIT_DIFF_MEAN",
    "PREV_DAYS_DECISION_MAX",
    "PREV_RATE_DOWN_PAYMENT_MEAN",
    "PREV_APP_CREDIT_RATIO_MEAN",
    "PREV_CREDIT_GOODS_RATIO_MEAN",
    "PREV_RECENT_365_COUNT",
    "PREV_RECENT_365_REFUSAL_RATE",
    "PREV_RECENT_365_APP_CREDIT_RATIO_MEAN",
    "PREV_LAST_REFUSED_FLAG",
    "PREV_LAST_APP_CREDIT_RATIO",
]
INSTALLMENTS_AGG_COLS = [
    "INST_RECORD_COUNT",
    "INST_MISSED_RATE",
    "INST_DPD_MEAN",
    "INST_DPD_MAX",
    "INST_PAYMENT_RATIO_MEAN",
    "INST_PAYMENT_RATIO_MIN",
    "INST_LATE_COUNT",
    "INST_PAYMENT_RATIO_STD",
    "INST_UNDERPAY_RATE",
    "INST_SEVERE_DPD_RATE",
    "INST_RECENT_365_DPD_MEAN",
    "INST_RECENT_365_DPD_MAX",
    "INST_RECENT_365_PAYMENT_RATIO_MEAN",
]
POS_CASH_AGG_COLS = [
    "POS_RECORD_COUNT",
    "POS_DPD_MEAN",
    "POS_DPD_MAX",
    "POS_DPD_DEF_MEAN",
    "POS_DPD_DEF_MAX",
    "POS_COMPLETED_RATE",
    "POS_ACTIVE_RATE",
    "POS_CNT_INSTALMENT_FUTURE_MEAN",
]
CREDIT_CARD_AGG_COLS = [
    "CC_RECORD_COUNT",
    "CC_BALANCE_MEAN",
    "CC_LIMIT_MEAN",
    "CC_UTILIZATION_MEAN",
    "CC_PAYMENT_RATIO_MEAN",
    "CC_DPD_MEAN",
    "CC_DPD_MAX",
    "CC_DRAWINGS_ATM_SUM",
    "CC_DRAWINGS_CURRENT_SUM",
]
ALL_AGGREGATE_FEATURE_COLS = (
    BUREAU_AGG_COLS
    + PREVIOUS_AGG_COLS
    + INSTALLMENTS_AGG_COLS
    + POS_CASH_AGG_COLS
    + CREDIT_CARD_AGG_COLS
)
AGGREGATE_FAMILY_COLS = {
    "BUREAU": BUREAU_AGG_COLS,
    "PREVIOUS_APPLICATION": PREVIOUS_AGG_COLS,
    "INSTALLMENTS": INSTALLMENTS_AGG_COLS,
    "POS_CASH": POS_CASH_AGG_COLS,
    "CREDIT_CARD": CREDIT_CARD_AGG_COLS,
}
DEFAULT_FULL_FEATURE_VIEW = "FULL_COMPLETE"
FULL_FEATURE_VIEWS = {
    "FULL_COMPLETE": ("BUREAU", "PREVIOUS_APPLICATION", "INSTALLMENTS", "POS_CASH", "CREDIT_CARD"),
    "FULL_NO_CREDIT_CARD": ("BUREAU", "PREVIOUS_APPLICATION", "INSTALLMENTS", "POS_CASH"),
    "FULL_NO_POS_CASH": ("BUREAU", "PREVIOUS_APPLICATION", "INSTALLMENTS", "CREDIT_CARD"),
    "FULL_NO_PREVIOUS_APPLICATION": ("BUREAU", "INSTALLMENTS", "POS_CASH", "CREDIT_CARD"),
    "FULL_NO_INSTALLMENTS": ("BUREAU", "PREVIOUS_APPLICATION", "POS_CASH", "CREDIT_CARD"),
    "FULL_NO_BUREAU": ("PREVIOUS_APPLICATION", "INSTALLMENTS", "POS_CASH", "CREDIT_CARD"),
}
FULL_FEATURE_BUILDER_ARTIFACT_PATH = "artifacts/full_feature_builder.joblib"
REDUCED_FEATURE_BUILDER_ARTIFACT_PATH = "artifacts/reduced_feature_builder.joblib"
FULL_FEATURE_BUILDER_MANIFEST_PATH = "artifacts/full_feature_builder.manifest.json"
REDUCED_FEATURE_BUILDER_MANIFEST_PATH = "artifacts/reduced_feature_builder.manifest.json"
FEATURE_ENGINEERING_VERSION = "feature-engineering-manifest-v2"
AGGREGATE_CONTRACT_VERSION = f"full_{len(ALL_AGGREGATE_FEATURE_COLS)}__reduced_0"
PROCESSED_SPLIT_NAMES = ["train", "val_model", "val_policy", "test"]
__all__ = [
    "FrozenFeatureBuilder",
    "fit_full_builder",
    "fit_reduced_builder",
    "build_full",
    "build_reduced",
    "pool_rare_categories",
    "safe_div",
    "assert_unique_key",
    "safe_left_merge_one_to_one",
    "prefix_columns",
    "resolve_full_feature_view",
]


@dataclass
class FrozenFeatureBuilder:
    tier: str
    rare_category_maps_: dict[str, set[str]] = field(default_factory=dict)
    flag_columns_: list[str] = field(default_factory=list)
    numeric_imputers_: dict[str, float] = field(default_factory=dict)
    categorical_fill_values_: dict[str, str] = field(default_factory=dict)
    pre_model_columns_: list[str] = field(default_factory=list)
    encoded_columns_: list[str] = field(default_factory=list)
    numeric_scale_columns_: list[str] = field(default_factory=list)
    aggregate_feature_cols_: list[str] = field(default_factory=list)
    categorical_columns_: list[str] = field(default_factory=list)
    scaler_: StandardScaler | None = None
    fit_row_count_: int = 0
    fit_split_name_: str = "train"
    dataset_fingerprint_: str = ""
    processed_manifest_fingerprint_: str | None = None
    feature_engineering_version_: str = FEATURE_ENGINEERING_VERSION
    aggregate_contract_version_: str = AGGREGATE_CONTRACT_VERSION

    def save(self, path: str) -> None:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        tmp_path = f"{path}.tmp"
        joblib.dump(self, tmp_path)
        os.replace(tmp_path, path)
        manifest_path = _builder_manifest_path(path)
        manifest = _build_builder_manifest(self)
        tmp_manifest_path = f"{manifest_path}.tmp"
        with open(tmp_manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
        os.replace(tmp_manifest_path, manifest_path)

    def transform(
        self,
        df: pd.DataFrame,
        for_linear_model: bool = False,
        raw_dir: str | None = None,
    ) -> pd.DataFrame:
        return _transform_with_builder(df, self, for_linear_model, raw_dir)




def _builder_manifest_path(builder_path: str) -> str:
    if builder_path.endswith(".joblib"):
        return builder_path[:-7] + ".manifest.json"
    return builder_path + ".manifest.json"


def _default_processed_manifest_path() -> str:
    return os.path.join(
        os.environ.get("DATA_PROCESSED_DIR", "data/processed/"),
        "processed_artifact_manifest.json",
    )


def _load_processed_manifest_fingerprint(processed_manifest_path: str | None = None) -> str | None:
    manifest_path = processed_manifest_path or _default_processed_manifest_path()
    if not os.path.exists(manifest_path):
        return None
    try:
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except Exception:
        return None
    if not isinstance(manifest, dict):
        return None
    fingerprint = manifest.get("processed_manifest_fingerprint")
    return str(fingerprint) if fingerprint else None


def _build_builder_manifest(builder: FrozenFeatureBuilder) -> dict[str, object]:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    return {
        "manifest_version": 2,
        "feature_engineering_version": builder.feature_engineering_version_,
        "git_commit": safe_git_commit(project_root),
        "fit_timestamp": datetime.now(timezone.utc).isoformat(),
        "builder_tier": builder.tier.upper(),
        "fit_row_count": int(builder.fit_row_count_),
        "fit_split_name": builder.fit_split_name_,
        "dataset_fingerprint": builder.dataset_fingerprint_,
        "processed_manifest_fingerprint": builder.processed_manifest_fingerprint_,
        "pre_model_column_schema_hash": column_sequence_hash(builder.pre_model_columns_),
        "encoded_column_count": int(len(builder.encoded_columns_)),
        "encoded_column_schema_hash": column_sequence_hash(builder.encoded_columns_),
        "rare_map_schema_hash": rare_map_schema_hash(builder.rare_category_maps_),
        "rare_map_sizes": {
            key: len(value) for key, value in sorted(builder.rare_category_maps_.items())
        },
        "categorical_columns": list(builder.categorical_columns_),
        "aggregate_contract_version": builder.aggregate_contract_version_,
    }

def pool_rare_categories(series: pd.Series, min_count: int = 500) -> pd.Series:
    vc = series.value_counts(dropna=False)
    keep = vc[vc >= min_count].index
    return series.where(series.isin(keep), other="OTHER")


def safe_div(a: pd.Series | np.ndarray, b: pd.Series | np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where((pd.notna(a)) & (pd.notna(b)) & (b != 0), a / b, np.nan)


def resolve_full_feature_view(feature_view: str | None) -> tuple[str, tuple[str, ...], list[str]]:
    view_name = DEFAULT_FULL_FEATURE_VIEW if feature_view is None else str(feature_view).upper()
    if view_name not in FULL_FEATURE_VIEWS:
        raise ValueError(f"Unsupported FULL feature view: {feature_view!r}")
    families = FULL_FEATURE_VIEWS[view_name]
    aggregate_feature_cols: list[str] = []
    for family in families:
        aggregate_feature_cols.extend(AGGREGATE_FAMILY_COLS[family])
    return view_name, families, aggregate_feature_cols


def _aggregate_contract_version_for_view(
    feature_view: str | None,
    aggregate_feature_cols: list[str],
) -> str:
    view_name, _, _ = resolve_full_feature_view(feature_view)
    if view_name == DEFAULT_FULL_FEATURE_VIEW and list(aggregate_feature_cols) == list(ALL_AGGREGATE_FEATURE_COLS):
        return AGGREGATE_CONTRACT_VERSION
    return f"{view_name.lower()}__full_{len(aggregate_feature_cols)}__reduced_0"


def assert_unique_key(df: pd.DataFrame, key: str, name: str) -> None:
    dupes = df[key].duplicated().sum()
    if dupes > 0:
        raise ValueError(f"{name} is not unique on {key}: {dupes} duplicate keys")


def safe_left_merge_one_to_one(
    base_df: pd.DataFrame,
    feat_df: pd.DataFrame,
    key: str = "SK_ID_CURR",
    feat_name: str = "features",
) -> pd.DataFrame:
    assert_unique_key(base_df, key, "base_df")
    assert_unique_key(feat_df, key, feat_name)
    return base_df.merge(feat_df, on=key, how="left", validate="one_to_one")


def prefix_columns(df: pd.DataFrame, prefix: str, key: str = "SK_ID_CURR") -> pd.DataFrame:
    cols = [c for c in df.columns if c != key]
    return df.rename(columns={c: f"{prefix}_{c}".upper() for c in cols})


def _engineer_application_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    assert "AMT_INCOME_TOTAL_CAPPED" in df.columns, "AMT_INCOME_TOTAL_CAPPED missing - Module 1 not applied"
    assert "DAYS_EMPLOYED_ANOM" in df.columns, "DAYS_EMPLOYED_ANOM missing - Module 1 Trap A not applied"
    df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365
    df["CREDIT_INCOME_RATIO"] = safe_div(df["AMT_CREDIT"], df["AMT_INCOME_TOTAL_CAPPED"] )
    df["ANNUITY_INCOME_RATIO"] = safe_div(df["AMT_ANNUITY"], df["AMT_INCOME_TOTAL_CAPPED"] )
    df["GOODS_CREDIT_RATIO"] = safe_div(df["AMT_GOODS_PRICE"], df["AMT_CREDIT"])
    df["CREDIT_TERM_RATIO"] = safe_div(df["AMT_ANNUITY"], df["AMT_CREDIT"])
    df["EMPLOYED_BIRTH_RATIO"] = safe_div(df["DAYS_EMPLOYED"], df["DAYS_BIRTH"])
    df["ID_PUBLISH_REG_RATIO"] = safe_div(df["DAYS_ID_PUBLISH"], df["DAYS_REGISTRATION"])
    df["EXT_SOURCE_MEAN"] = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].mean(axis=1)
    df["EXT_SOURCE_STD"] = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].std(axis=1)
    df["SOCIAL_CIRCLE_SUM"] = (
        df["OBS_30_CNT_SOCIAL_CIRCLE"].fillna(0)
        + df["DEF_30_CNT_SOCIAL_CIRCLE"].fillna(0)
        + df["OBS_60_CNT_SOCIAL_CIRCLE"].fillna(0)
        + df["DEF_60_CNT_SOCIAL_CIRCLE"].fillna(0)
    )
    req_cols = [
        "AMT_REQ_CREDIT_BUREAU_HOUR",
        "AMT_REQ_CREDIT_BUREAU_DAY",
        "AMT_REQ_CREDIT_BUREAU_WEEK",
        "AMT_REQ_CREDIT_BUREAU_MON",
        "AMT_REQ_CREDIT_BUREAU_QRT",
        "AMT_REQ_CREDIT_BUREAU_YEAR",
    ]
    df["BUREAU_REQUEST_SUM"] = df[req_cols].fillna(0).sum(axis=1)
    return df


def _agg_bureau_balance(bureau_df: pd.DataFrame, raw_dir: str) -> pd.DataFrame:
    bb = pd.read_csv(
        os.path.join(raw_dir, "bureau_balance.csv"),
        usecols=["SK_ID_BUREAU", "MONTHS_BALANCE", "STATUS"],
    ).copy()
    bb["BB_STATUS_SEVERITY"] = bb["STATUS"].map({
        "X": np.nan,
        "C": 0.0,
        "0": 0.0,
        "1": 1.0,
        "2": 2.0,
        "3": 3.0,
        "4": 4.0,
        "5": 5.0,
    }).astype(float)
    bb["BB_ADVERSE_FLAG"] = bb["STATUS"].isin(["1", "2", "3", "4", "5"]).astype("int8")
    g = bb.groupby("SK_ID_BUREAU")
    out = pd.DataFrame(index=g.size().index)
    out["BB_MONTHS_COUNT"] = g.size()
    out["BB_ACCOUNT_MAX_STATUS_SEVERITY"] = g["BB_STATUS_SEVERITY"].max()
    out["BB_ACCOUNT_ADVERSE_RATE"] = g["BB_ADVERSE_FLAG"].mean()
    account_level = out.reset_index()
    bureau_keys = bureau_df[["SK_ID_BUREAU", "SK_ID_CURR"]].drop_duplicates()
    account_level = account_level.merge(
        bureau_keys,
        on="SK_ID_BUREAU",
        how="inner",
        validate="one_to_one",
    )
    current_level = account_level.groupby("SK_ID_CURR")
    out = pd.DataFrame(index=current_level.size().index)
    out["BB_MONTHS_COUNT_MEAN"] = current_level["BB_MONTHS_COUNT"].mean()
    out["BB_MAX_STATUS_MAX"] = current_level["BB_ACCOUNT_MAX_STATUS_SEVERITY"].max()
    out["BB_MAX_STATUS_MEAN"] = current_level["BB_ACCOUNT_MAX_STATUS_SEVERITY"].mean()
    out["BB_ADVERSE_ACCOUNT_RATE"] = current_level["BB_ACCOUNT_ADVERSE_RATE"].apply(
        lambda s: (s.fillna(0) > 0).mean()
    )
    out["BB_ADVERSE_RATE_MEAN"] = current_level["BB_ACCOUNT_ADVERSE_RATE"].mean()
    result = out.reset_index()
    assert_unique_key(result, "SK_ID_CURR", "bureau_balance_agg")
    return result


def _agg_bureau(raw_dir: str) -> pd.DataFrame:
    bureau = pd.read_csv(
        os.path.join(raw_dir, "bureau.csv"),
        usecols=[
            "SK_ID_CURR",
            "SK_ID_BUREAU",
            "CREDIT_ACTIVE",
            "DAYS_CREDIT",
            "CREDIT_DAY_OVERDUE",
            "AMT_CREDIT_MAX_OVERDUE",
            "CNT_CREDIT_PROLONG",
            "AMT_CREDIT_SUM",
            "AMT_CREDIT_SUM_DEBT",
            "AMT_CREDIT_SUM_OVERDUE",
        ],
    ).copy()
    g = bureau.groupby("SK_ID_CURR")
    out = pd.DataFrame(index=g.size().index)
    out["BUREAU_LOAN_COUNT"] = g.size()
    out["BUREAU_ACTIVE_COUNT"] = g["CREDIT_ACTIVE"].apply(lambda s: (s == "Active").sum())
    out["BUREAU_CLOSED_COUNT"] = g["CREDIT_ACTIVE"].apply(lambda s: (s == "Closed").sum())
    out["BUREAU_AMT_CREDIT_SUM_SUM"] = g["AMT_CREDIT_SUM"].sum()
    out["BUREAU_AMT_CREDIT_SUM_DEBT_SUM"] = g["AMT_CREDIT_SUM_DEBT"].sum()
    out["BUREAU_DEBT_TO_CREDIT_RATIO"] = safe_div(
        out["BUREAU_AMT_CREDIT_SUM_DEBT_SUM"].to_numpy(),
        out["BUREAU_AMT_CREDIT_SUM_SUM"].to_numpy(),
    )
    out["BUREAU_AMT_CREDIT_SUM_OVERDUE_SUM"] = g["AMT_CREDIT_SUM_OVERDUE"].sum()
    out["BUREAU_CREDIT_DAY_OVERDUE_MAX"] = g["CREDIT_DAY_OVERDUE"].max()
    out["BUREAU_DAYS_CREDIT_MAX"] = g["DAYS_CREDIT"].max()
    out["BUREAU_CNT_CREDIT_PROLONG_SUM"] = g["CNT_CREDIT_PROLONG"].sum()
    out["BUREAU_ACTIVE_SHARE"] = safe_div(
        out["BUREAU_ACTIVE_COUNT"].to_numpy(),
        out["BUREAU_LOAN_COUNT"].to_numpy(),
    )
    out["BUREAU_AMT_CREDIT_MAX_OVERDUE_MAX"] = g["AMT_CREDIT_MAX_OVERDUE"].max()
    active_credit = bureau.assign(
        _ACTIVE_CREDIT_SUM=np.where(
            bureau["CREDIT_ACTIVE"] == "Active",
            bureau["AMT_CREDIT_SUM"],
            np.nan,
        )
    ).groupby("SK_ID_CURR")["_ACTIVE_CREDIT_SUM"].sum(min_count=1)
    active_debt = bureau.assign(
        _ACTIVE_DEBT_SUM=np.where(
            bureau["CREDIT_ACTIVE"] == "Active",
            bureau["AMT_CREDIT_SUM_DEBT"],
            np.nan,
        )
    ).groupby("SK_ID_CURR")["_ACTIVE_DEBT_SUM"].sum(min_count=1)
    out["BUREAU_ACTIVE_DEBT_RATIO"] = safe_div(
        active_debt.reindex(out.index).to_numpy(),
        active_credit.reindex(out.index).to_numpy(),
    )
    out["BUREAU_DAYS_CREDIT_MEAN"] = g["DAYS_CREDIT"].mean()
    out["BUREAU_DAYS_CREDIT_MIN"] = g["DAYS_CREDIT"].min()
    result = out.reset_index()
    result = safe_left_merge_one_to_one(
        result,
        _agg_bureau_balance(bureau, raw_dir),
        "SK_ID_CURR",
        "bureau_balance_agg",
    )
    assert_unique_key(result, "SK_ID_CURR", "bureau_agg")
    return result


def _agg_previous(raw_dir: str) -> pd.DataFrame:
    prev = pd.read_csv(os.path.join(raw_dir, "previous_application.csv")).copy()
    prev["PREV_APP_CREDIT_DIFF_ROW"] = prev["AMT_APPLICATION"] - prev["AMT_CREDIT"]
    prev["PREV_APP_CREDIT_RATIO_ROW"] = np.where(
        prev["AMT_CREDIT"].notna() & (prev["AMT_CREDIT"] != 0),
        safe_div(prev["AMT_APPLICATION"], prev["AMT_CREDIT"]),
        np.nan,
    )
    prev["PREV_CREDIT_GOODS_RATIO_ROW"] = np.where(
        prev["AMT_GOODS_PRICE"].notna() & (prev["AMT_GOODS_PRICE"] != 0),
        safe_div(prev["AMT_CREDIT"], prev["AMT_GOODS_PRICE"]),
        np.nan,
    )
    prev["PREV_REFUSED_FLAG"] = (prev["NAME_CONTRACT_STATUS"] == "Refused").astype("int8")
    prev["PREV_RECENT_365_FLAG"] = (prev["DAYS_DECISION"] >= -365).astype("int8")
    prev["PREV_RECENT_365_REFUSED_FLAG"] = np.where(
        prev["DAYS_DECISION"] >= -365,
        prev["PREV_REFUSED_FLAG"],
        np.nan,
    )
    prev["PREV_RECENT_365_APP_CREDIT_RATIO"] = np.where(
        prev["DAYS_DECISION"] >= -365,
        prev["PREV_APP_CREDIT_RATIO_ROW"],
        np.nan,
    )
    g = prev.groupby("SK_ID_CURR")
    app_count = g.size()
    approved = g["NAME_CONTRACT_STATUS"].apply(lambda s: (s == "Approved").sum())
    refused = g["NAME_CONTRACT_STATUS"].apply(lambda s: (s == "Refused").sum())
    out = pd.DataFrame(index=app_count.index)
    out["PREV_APP_COUNT"] = app_count
    out["PREV_APPROVED_COUNT"] = approved
    out["PREV_REFUSED_COUNT"] = refused
    out["PREV_APPROVAL_RATE"] = safe_div(approved.to_numpy(), app_count.to_numpy())
    out["PREV_REFUSAL_RATE"] = safe_div(refused.to_numpy(), app_count.to_numpy())
    out["PREV_AMT_APPLICATION_MEAN"] = g["AMT_APPLICATION"].mean()
    out["PREV_AMT_CREDIT_MEAN"] = g["AMT_CREDIT"].mean()
    out["PREV_AMT_GOODS_PRICE_MEAN"] = g["AMT_GOODS_PRICE"].mean()
    out["PREV_APP_CREDIT_DIFF_MEAN"] = g["PREV_APP_CREDIT_DIFF_ROW"].mean()
    out["PREV_DAYS_DECISION_MAX"] = g["DAYS_DECISION"].max()
    out["PREV_RATE_DOWN_PAYMENT_MEAN"] = g["RATE_DOWN_PAYMENT"].mean()
    out["PREV_APP_CREDIT_RATIO_MEAN"] = g["PREV_APP_CREDIT_RATIO_ROW"].mean()
    out["PREV_CREDIT_GOODS_RATIO_MEAN"] = g["PREV_CREDIT_GOODS_RATIO_ROW"].mean()
    out["PREV_RECENT_365_COUNT"] = g["PREV_RECENT_365_FLAG"].sum()
    out["PREV_RECENT_365_REFUSAL_RATE"] = g["PREV_RECENT_365_REFUSED_FLAG"].mean()
    out["PREV_RECENT_365_APP_CREDIT_RATIO_MEAN"] = g["PREV_RECENT_365_APP_CREDIT_RATIO"].mean()
    last_idx = prev.groupby("SK_ID_CURR")["DAYS_DECISION"].idxmax()
    last_rows = prev.loc[
        last_idx,
        ["SK_ID_CURR", "PREV_REFUSED_FLAG", "PREV_APP_CREDIT_RATIO_ROW"],
    ].rename(
        columns={
            "PREV_REFUSED_FLAG": "PREV_LAST_REFUSED_FLAG",
            "PREV_APP_CREDIT_RATIO_ROW": "PREV_LAST_APP_CREDIT_RATIO",
        }
    )
    out = out.reset_index().merge(last_rows, on="SK_ID_CURR", how="left", validate="one_to_one")
    assert_unique_key(out, "SK_ID_CURR", "previous_app_agg")
    return out


def _normalize_child_merge_curr(df: pd.DataFrame, label: str) -> pd.DataFrame:
    if {"SK_ID_CURR_x", "SK_ID_CURR_y"}.issubset(df.columns):
        if not df["SK_ID_CURR_x"].equals(df["SK_ID_CURR_y"]):
            raise ValueError(f"{label} merge produced mismatched SK_ID_CURR values")
        df["SK_ID_CURR"] = df["SK_ID_CURR_y"]
        df = df.drop(columns=["SK_ID_CURR_x", "SK_ID_CURR_y"])
    return df


def _agg_installments(raw_dir: str) -> pd.DataFrame:
    inst = pd.read_csv(os.path.join(raw_dir, "installments_payments.csv"))
    prev_keys = pd.read_csv(
        os.path.join(raw_dir, "previous_application.csv"),
        usecols=["SK_ID_PREV", "SK_ID_CURR"],
    ).drop_duplicates()
    inst = inst.merge(prev_keys, on="SK_ID_PREV", how="inner", validate="many_to_one")
    inst = _normalize_child_merge_curr(inst, "installments").copy()
    inst["INST_MISSED_FLAG"] = inst["DAYS_ENTRY_PAYMENT"].isna().astype("int8")
    inst["INST_DPD"] = np.where(
        inst["DAYS_ENTRY_PAYMENT"].notna(),
        np.maximum(inst["DAYS_ENTRY_PAYMENT"] - inst["DAYS_INSTALMENT"], 0),
        np.nan,
    )
    inst["INST_PAYMENT_RATIO"] = np.where(
        (inst["AMT_INSTALMENT"] > 0) & inst["AMT_PAYMENT"].notna(),
        safe_div(inst["AMT_PAYMENT"], inst["AMT_INSTALMENT"]),
        np.nan,
    )
    inst["INST_UNDERPAY_FLAG"] = np.where(
        inst["INST_PAYMENT_RATIO"].notna(),
        (inst["INST_PAYMENT_RATIO"] < 0.98).astype("int8"),
        np.nan,
    )
    inst["INST_SEVERE_DPD_FLAG"] = np.where(
        inst["INST_DPD"].notna(),
        (inst["INST_DPD"] >= 30).astype("int8"),
        np.nan,
    )
    inst["INST_RECENT_365_DPD"] = np.where(
        inst["DAYS_INSTALMENT"] >= -365,
        inst["INST_DPD"],
        np.nan,
    )
    inst["INST_RECENT_365_PAYMENT_RATIO"] = np.where(
        inst["DAYS_INSTALMENT"] >= -365,
        inst["INST_PAYMENT_RATIO"],
        np.nan,
    )
    g = inst.groupby("SK_ID_CURR")
    out = pd.DataFrame(index=g.size().index)
    out["INST_RECORD_COUNT"] = g.size()
    out["INST_MISSED_RATE"] = g["INST_MISSED_FLAG"].mean()
    out["INST_DPD_MEAN"] = g["INST_DPD"].mean()
    out["INST_DPD_MAX"] = g["INST_DPD"].max()
    out["INST_PAYMENT_RATIO_MEAN"] = g["INST_PAYMENT_RATIO"].mean()
    out["INST_PAYMENT_RATIO_MIN"] = g["INST_PAYMENT_RATIO"].min()
    out["INST_LATE_COUNT"] = g["INST_DPD"].apply(lambda s: (s > 0).sum())
    out["INST_PAYMENT_RATIO_STD"] = g["INST_PAYMENT_RATIO"].std()
    out["INST_UNDERPAY_RATE"] = g["INST_UNDERPAY_FLAG"].mean()
    out["INST_SEVERE_DPD_RATE"] = g["INST_SEVERE_DPD_FLAG"].mean()
    out["INST_RECENT_365_DPD_MEAN"] = g["INST_RECENT_365_DPD"].mean()
    out["INST_RECENT_365_DPD_MAX"] = g["INST_RECENT_365_DPD"].max()
    out["INST_RECENT_365_PAYMENT_RATIO_MEAN"] = g["INST_RECENT_365_PAYMENT_RATIO"].mean()
    result = out.reset_index()
    assert_unique_key(result, "SK_ID_CURR", "installments_agg")
    return result


def _agg_pos_cash(raw_dir: str) -> pd.DataFrame:
    pos = pd.read_csv(os.path.join(raw_dir, "POS_CASH_balance.csv"))
    prev_keys = pd.read_csv(
        os.path.join(raw_dir, "previous_application.csv"),
        usecols=["SK_ID_PREV", "SK_ID_CURR"],
    ).drop_duplicates()
    pos = pos.merge(prev_keys, on="SK_ID_PREV", how="inner", validate="many_to_one")
    pos = _normalize_child_merge_curr(pos, "pos cash")
    g = pos.groupby("SK_ID_CURR")
    out = pd.DataFrame(index=g.size().index)
    out["POS_RECORD_COUNT"] = g.size()
    out["POS_DPD_MEAN"] = g["SK_DPD"].mean()
    out["POS_DPD_MAX"] = g["SK_DPD"].max()
    out["POS_DPD_DEF_MEAN"] = g["SK_DPD_DEF"].mean()
    out["POS_DPD_DEF_MAX"] = g["SK_DPD_DEF"].max()
    out["POS_COMPLETED_RATE"] = g["NAME_CONTRACT_STATUS"].apply(lambda s: (s == "Completed").mean())
    out["POS_ACTIVE_RATE"] = g["NAME_CONTRACT_STATUS"].apply(lambda s: (s == "Active").mean())
    out["POS_CNT_INSTALMENT_FUTURE_MEAN"] = g["CNT_INSTALMENT_FUTURE"].mean()
    result = out.reset_index()
    assert_unique_key(result, "SK_ID_CURR", "pos_cash_agg")
    return result


def _agg_credit_card(raw_dir: str) -> pd.DataFrame:
    cc = pd.read_csv(os.path.join(raw_dir, "credit_card_balance.csv"))
    prev_keys = pd.read_csv(
        os.path.join(raw_dir, "previous_application.csv"),
        usecols=["SK_ID_PREV", "SK_ID_CURR"],
    ).drop_duplicates()
    cc = cc.merge(prev_keys, on="SK_ID_PREV", how="inner", validate="many_to_one")
    cc = _normalize_child_merge_curr(cc, "credit card").copy()
    cc["CC_UTILIZATION_ROW"] = np.where(
        cc["AMT_CREDIT_LIMIT_ACTUAL"] > 0,
        safe_div(cc["AMT_BALANCE"], cc["AMT_CREDIT_LIMIT_ACTUAL"]),
        np.nan,
    )
    cc["CC_PAYMENT_RATIO_ROW"] = np.where(
        cc["AMT_INST_MIN_REGULARITY"].notna() & (cc["AMT_INST_MIN_REGULARITY"] > 0),
        safe_div(cc["AMT_PAYMENT_TOTAL_CURRENT"], cc["AMT_INST_MIN_REGULARITY"]),
        np.nan,
    )
    g = cc.groupby("SK_ID_CURR")
    out = pd.DataFrame(index=g.size().index)
    out["CC_RECORD_COUNT"] = g.size()
    out["CC_BALANCE_MEAN"] = g["AMT_BALANCE"].mean()
    out["CC_LIMIT_MEAN"] = g["AMT_CREDIT_LIMIT_ACTUAL"].mean()
    out["CC_UTILIZATION_MEAN"] = g["CC_UTILIZATION_ROW"].mean()
    out["CC_PAYMENT_RATIO_MEAN"] = g["CC_PAYMENT_RATIO_ROW"].mean()
    out["CC_DPD_MEAN"] = g["SK_DPD"].mean()
    out["CC_DPD_MAX"] = g["SK_DPD"].max()
    out["CC_DRAWINGS_ATM_SUM"] = g["AMT_DRAWINGS_ATM_CURRENT"].sum()
    out["CC_DRAWINGS_CURRENT_SUM"] = g["AMT_DRAWINGS_CURRENT"].sum()
    result = out.reset_index()
    assert_unique_key(result, "SK_ID_CURR", "credit_card_agg")
    return result


def _processed_split_path(split_name: str) -> str:
    return os.path.join(
        os.environ.get("DATA_PROCESSED_DIR", "data/processed/"),
        f"{split_name}.pkl",
    )


def _validate_builder(builder: FrozenFeatureBuilder, expected_tier: str) -> None:
    if not isinstance(builder, FrozenFeatureBuilder):
        raise TypeError("builder must be a FrozenFeatureBuilder instance")
    if builder.tier.upper() != expected_tier.upper():
        raise ValueError(
            f"{expected_tier.upper()} transform requires a "
            f"{expected_tier.upper()} FrozenFeatureBuilder"
        )


def _select_application_frame(df: pd.DataFrame, require_sk_id_curr: bool) -> pd.DataFrame:
    missing = [c for c in APPLICATION_REQUIRED_INPUT_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required application columns: {sorted(missing)}")
    cols = ["SK_ID_CURR"] if require_sk_id_curr else []
    cols += APPLICATION_REQUIRED_INPUT_COLS
    cols += [c for c in FAIRNESS_ONLY_COLS if c in df.columns]
    return df[cols].copy()


def _merge_full_aggregates(
    base_df: pd.DataFrame,
    raw_dir: str,
    feature_view: str = DEFAULT_FULL_FEATURE_VIEW,
) -> pd.DataFrame:
    _, families, _ = resolve_full_feature_view(feature_view)
    merged = base_df
    family_frames: list[tuple[str, pd.DataFrame]] = []
    if "BUREAU" in families:
        family_frames.append(("bureau_agg", _agg_bureau(raw_dir)))
    if "PREVIOUS_APPLICATION" in families:
        family_frames.append(("previous_app_agg", _agg_previous(raw_dir)))
    if "INSTALLMENTS" in families:
        family_frames.append(("installments_agg", _agg_installments(raw_dir)))
    if "POS_CASH" in families:
        family_frames.append(("pos_cash_agg", _agg_pos_cash(raw_dir)))
    if "CREDIT_CARD" in families:
        family_frames.append(("credit_card_agg", _agg_credit_card(raw_dir)))
    for feat_name, feat_df in family_frames:
        merged = safe_left_merge_one_to_one(merged, feat_df, "SK_ID_CURR", feat_name)
    return merged


def _build_pre_model_frame(
    df: pd.DataFrame,
    tier: str,
    raw_dir: str | None = None,
    allow_flattened_full_input: bool = True,
    feature_view: str = DEFAULT_FULL_FEATURE_VIEW,
) -> pd.DataFrame:
    tier = tier.upper()
    selected_agg_cols: list[str] = []
    has_any_aggs = False
    if tier == "FULL":
        _, _, selected_agg_cols = resolve_full_feature_view(feature_view)
        has_any_aggs = any(c in df.columns for c in selected_agg_cols)
    use_flat_aggs = tier == "FULL" and allow_flattened_full_input and has_any_aggs
    base = _select_application_frame(df, require_sk_id_curr=(tier == "FULL" and not use_flat_aggs))
    base = _engineer_application_features(base)
    if tier == "FULL":
        if use_flat_aggs:
            flat_agg_cols = [c for c in selected_agg_cols if c in df.columns]
            base = pd.concat(
                [base.reset_index(drop=True), df[flat_agg_cols].reset_index(drop=True)],
                axis=1,
            )
        else:
            if raw_dir is None:
                raise ValueError("FULL transform requires flattened aggregate columns or raw_dir with SK_ID_CURR")
            base = _merge_full_aggregates(base, raw_dir, feature_view=feature_view)
    if "SK_ID_CURR" in base.columns:
        base = base.drop(columns=["SK_ID_CURR"])
    fairness_cols = [c for c in FAIRNESS_ONLY_COLS if c in base.columns]
    if fairness_cols:
        base = base.drop(columns=fairness_cols)
    return base


def _fit_rare_category_maps(
    df: pd.DataFrame,
    categorical_cols: list[str],
    min_count: int = 500,
) -> dict[str, set[str]]:
    maps: dict[str, set[str]] = {}
    for col in categorical_cols:
        series = df[col].fillna("MISSING")
        vc = series.value_counts(dropna=False)
        maps[col] = set(vc[vc >= min_count].index.tolist())
    return maps


def _apply_rare_category_maps(
    df: pd.DataFrame,
    categorical_cols: list[str],
    fill_values: dict[str, str],
    rare_maps: dict[str, set[str]],
) -> pd.DataFrame:
    df = df.copy()
    for col in categorical_cols:
        if col not in df.columns:
            continue
        series = df[col].fillna(fill_values.get(col, "MISSING"))
        df[col] = series.where(series.isin(rare_maps.get(col, set())), other="OTHER")
    return df


def _fit_missing_flag_columns(df: pd.DataFrame) -> list[str]:
    miss_rate = df.isna().mean()
    return miss_rate[miss_rate >= 0.05].index.tolist()


def _create_missing_flag_columns(df: pd.DataFrame, flag_cols: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in flag_cols:
        if col in df.columns:
            df[f"{col}_IS_MISSING"] = df[col].isna().astype("int8")
    return df


def _fit_numeric_imputers(df: pd.DataFrame, numeric_cols: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for col in numeric_cols:
        median = df[col].median(skipna=True)
        out[col] = 0.0 if pd.isna(median) else float(median)
    return out


def _prepare_linear_numeric_frame(
    df: pd.DataFrame,
    numeric_imputers: dict[str, float],
) -> pd.DataFrame:
    df = df.copy()
    for col, value in numeric_imputers.items():
        if col in df.columns:
            df[col] = df[col].fillna(value)
    return df


def _fit_builder_from_pre_model_frame(
    train_pre_model_df: pd.DataFrame,
    tier: str,
    aggregate_feature_cols: list[str],
    *,
    dataset_fingerprint: str | None = None,
    fit_split_name: str = "train",
    processed_manifest_fingerprint: str | None = None,
    aggregate_contract_version: str = AGGREGATE_CONTRACT_VERSION,
) -> FrozenFeatureBuilder:
    train = train_pre_model_df.copy()
    categorical_cols = [c for c in CATEGORICAL_MODEL_COLS if c in train.columns]
    fill_values = {col: "MISSING" for col in categorical_cols}
    pre_model_columns = train.columns.tolist()
    rare_maps = _fit_rare_category_maps(train, categorical_cols)
    train = _apply_rare_category_maps(train, categorical_cols, fill_values, rare_maps)
    flag_cols = _fit_missing_flag_columns(train)
    train = _create_missing_flag_columns(train, flag_cols)
    numeric_scale_cols = [c for c in train.select_dtypes(include=[np.number]).columns if not c.endswith("_IS_MISSING")]
    numeric_imputers = _fit_numeric_imputers(train, numeric_scale_cols)
    linear_ready = _prepare_linear_numeric_frame(train, numeric_imputers)
    encoded_train = pd.get_dummies(linear_ready, columns=categorical_cols, dtype=float, drop_first=False)
    scaler = None
    if numeric_scale_cols:
        scaler = StandardScaler(with_mean=False)
        scaler.fit(encoded_train[numeric_scale_cols])
    return FrozenFeatureBuilder(
        tier=tier.upper(),
        rare_category_maps_=rare_maps,
        flag_columns_=flag_cols,
        numeric_imputers_=numeric_imputers,
        categorical_fill_values_=fill_values,
        pre_model_columns_=pre_model_columns,
        encoded_columns_=encoded_train.columns.tolist(),
        numeric_scale_columns_=numeric_scale_cols,
        aggregate_feature_cols_=aggregate_feature_cols,
        categorical_columns_=categorical_cols,
        scaler_=scaler,
        fit_row_count_=int(len(train_pre_model_df)),
        fit_split_name_=fit_split_name,
        dataset_fingerprint_=dataset_fingerprint or dataframe_fingerprint(train_pre_model_df),
        processed_manifest_fingerprint_=processed_manifest_fingerprint,
        feature_engineering_version_=FEATURE_ENGINEERING_VERSION,
        aggregate_contract_version_=aggregate_contract_version,
    )


def fit_full_builder(
    train_df: pd.DataFrame,
    raw_dir: str = "data/raw/",
    save_path: str | None = None,
    processed_manifest_path: str | None = None,
    feature_view: str = DEFAULT_FULL_FEATURE_VIEW,
) -> FrozenFeatureBuilder:
    view_name, _, aggregate_feature_cols = resolve_full_feature_view(feature_view)
    builder = _fit_builder_from_pre_model_frame(
        _build_pre_model_frame(
            train_df,
            "FULL",
            raw_dir=raw_dir,
            allow_flattened_full_input=False,
            feature_view=view_name,
        ),
        "FULL",
        aggregate_feature_cols,
        dataset_fingerprint=dataframe_fingerprint(train_df),
        fit_split_name="train",
        processed_manifest_fingerprint=_load_processed_manifest_fingerprint(processed_manifest_path),
        aggregate_contract_version=_aggregate_contract_version_for_view(view_name, aggregate_feature_cols),
    )
    if save_path is not None:
        builder.save(save_path)
    return builder


def fit_reduced_builder(
    train_df: pd.DataFrame,
    save_path: str | None = None,
    processed_manifest_path: str | None = None,
) -> FrozenFeatureBuilder:
    builder = _fit_builder_from_pre_model_frame(
        _build_pre_model_frame(train_df, "REDUCED", raw_dir=None, allow_flattened_full_input=False),
        "REDUCED",
        [],
        dataset_fingerprint=dataframe_fingerprint(train_df),
        fit_split_name="train",
        processed_manifest_fingerprint=_load_processed_manifest_fingerprint(processed_manifest_path),
    )
    if save_path is not None:
        builder.save(save_path)
    return builder


def _align_pre_model_frame(df: pd.DataFrame, expected_columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in expected_columns:
        if col not in df.columns:
            df[col] = np.nan
    extra = [c for c in df.columns if c not in expected_columns]
    if extra:
        df = df.drop(columns=extra)
    return df[expected_columns]


def _encode_to_frozen_columns(df: pd.DataFrame, builder: FrozenFeatureBuilder) -> pd.DataFrame:
    encoded = pd.get_dummies(df, columns=builder.categorical_columns_, dtype=float, drop_first=False)
    for col in builder.encoded_columns_:
        if col not in encoded.columns:
            encoded[col] = 0.0
    extra = [c for c in encoded.columns if c not in builder.encoded_columns_]
    if extra:
        encoded = encoded.drop(columns=extra)
    return encoded[builder.encoded_columns_]


def _scale_frozen_numeric_columns(df: pd.DataFrame, builder: FrozenFeatureBuilder) -> pd.DataFrame:
    if builder.scaler_ is None or not builder.numeric_scale_columns_:
        return df
    df = df.copy()
    df[builder.numeric_scale_columns_] = builder.scaler_.transform(df[builder.numeric_scale_columns_])
    return df


def _transform_with_builder(
    df: pd.DataFrame,
    builder: FrozenFeatureBuilder,
    for_linear_model: bool = False,
    raw_dir: str | None = None,
    feature_view: str = DEFAULT_FULL_FEATURE_VIEW,
) -> pd.DataFrame:
    pre_model = _build_pre_model_frame(
        df,
        builder.tier,
        raw_dir=raw_dir,
        allow_flattened_full_input=True,
        feature_view=feature_view,
    )
    pre_model = _align_pre_model_frame(pre_model, builder.pre_model_columns_)
    if builder.tier == "FULL" and raw_dir is None:
        for col, value in builder.numeric_imputers_.items():
            if col in pre_model.columns and pre_model[col].isna().all():
                pre_model[col] = value
    pre_model = _apply_rare_category_maps(
        pre_model,
        builder.categorical_columns_,
        builder.categorical_fill_values_,
        builder.rare_category_maps_,
    )
    pre_model = _create_missing_flag_columns(pre_model, builder.flag_columns_)
    if for_linear_model:
        pre_model = _prepare_linear_numeric_frame(pre_model, builder.numeric_imputers_)
    encoded = _encode_to_frozen_columns(pre_model, builder)
    if for_linear_model:
        encoded = _scale_frozen_numeric_columns(encoded, builder)
    return encoded


def build_full(
    df: pd.DataFrame,
    builder: FrozenFeatureBuilder,
    for_linear_model: bool = False,
    raw_dir: str | None = None,
    feature_view: str = DEFAULT_FULL_FEATURE_VIEW,
) -> pd.DataFrame:
    _validate_builder(builder, "FULL")
    return _transform_with_builder(df, builder, for_linear_model, raw_dir, feature_view)


def build_reduced(
    df: pd.DataFrame,
    builder: FrozenFeatureBuilder,
    for_linear_model: bool = False,
) -> pd.DataFrame:
    _validate_builder(builder, "REDUCED")
    return _transform_with_builder(df, builder, for_linear_model, None)


def _assert_blocked_output_columns(output_df: pd.DataFrame) -> None:
    blocked_exact = set(FORBIDDEN_COLS) | set(FAIRNESS_ONLY_COLS)
    blocked_flags = {f"{col}_IS_MISSING" for col in FAIRNESS_ONLY_COLS}
    found = [c for c in output_df.columns if c in blocked_exact or c in blocked_flags]
    assert not found, f"Blocked columns leaked into output: {found}"


if __name__ == "__main__":
    print("=" * 60)
    print("Stage 1 - Utility Function Unit Tests")
    print("=" * 60)
    s = pd.Series(["A"] * 600 + ["B"] * 600 + ["C"] * 10)
    pooled = pool_rare_categories(s, min_count=500)
    assert (pooled == "OTHER").sum() == 10
    assert (pooled == "A").sum() == 600
    assert (pooled == "B").sum() == 600
    assert (s == "C").sum() == 10
    print("  pool_rare_categories ... PASSED")
    a = pd.Series([4.0, np.nan, 6.0])
    b = pd.Series([2.0, 2.0, 0.0])
    result = safe_div(a, b)
    np.testing.assert_array_equal(np.where(np.isnan(result), -999, result), np.array([2.0, -999, -999]))
    print("  safe_div ................. PASSED")
    df_ok = pd.DataFrame({"K": [1, 2, 3], "V": [10, 20, 30]})
    assert_unique_key(df_ok, "K", "test_ok")
    df_dup = pd.DataFrame({"K": [1, 1, 3], "V": [10, 20, 30]})
    try:
        assert_unique_key(df_dup, "K", "test_dup")
        assert False
    except ValueError as exc:
        assert "duplicate keys" in str(exc)
    print("  assert_unique_key ........ PASSED")
    base = pd.DataFrame({"SK_ID_CURR": [1, 2, 3], "A": [10, 20, 30]})
    feat = pd.DataFrame({"SK_ID_CURR": [1, 2, 3], "B": [100, 200, 300]})
    merged = safe_left_merge_one_to_one(base, feat)
    assert list(merged.columns) == ["SK_ID_CURR", "A", "B"]
    feat_dup = pd.DataFrame({"SK_ID_CURR": [1, 1, 3], "B": [100, 200, 300]})
    try:
        safe_left_merge_one_to_one(base, feat_dup)
        assert False
    except ValueError as exc:
        assert "duplicate keys" in str(exc)
    print("  safe_left_merge_one_to_one PASSED")
    renamed = prefix_columns(pd.DataFrame({"SK_ID_CURR": [1, 2], "val_a": [10, 20]}), "TEST")
    assert "SK_ID_CURR" in renamed.columns and "TEST_VAL_A" in renamed.columns
    print("  prefix_columns ........... PASSED")
    print("All Stage 1 utility tests PASSED\n")

    print("=" * 60)
    print("Stage 2 - Application Feature Engineering Tests")
    print("=" * 60)
    rng = np.random.default_rng(42)
    synth_df = pd.DataFrame(
        {
            "AMT_INCOME_TOTAL_CAPPED": rng.uniform(50_000, 500_000, 10),
            "AMT_CREDIT": rng.uniform(100_000, 1_000_000, 10),
            "AMT_ANNUITY": rng.uniform(5_000, 50_000, 10),
            "AMT_GOODS_PRICE": rng.uniform(50_000, 800_000, 10),
            "DAYS_BIRTH": rng.uniform(-25_000, -7_000, 10),
            "DAYS_EMPLOYED": np.where(rng.random(10) > 0.2, rng.uniform(-5_000, -100, 10), np.nan),
            "DAYS_REGISTRATION": rng.uniform(-15_000, -500, 10),
            "DAYS_ID_PUBLISH": rng.uniform(-6_000, -100, 10),
            "EXT_SOURCE_1": rng.uniform(0, 1, 10),
            "EXT_SOURCE_2": rng.uniform(0, 1, 10),
            "EXT_SOURCE_3": rng.uniform(0, 1, 10),
            "OBS_30_CNT_SOCIAL_CIRCLE": rng.integers(0, 5, 10).astype(float),
            "DEF_30_CNT_SOCIAL_CIRCLE": rng.integers(0, 3, 10).astype(float),
            "OBS_60_CNT_SOCIAL_CIRCLE": rng.integers(0, 5, 10).astype(float),
            "DEF_60_CNT_SOCIAL_CIRCLE": rng.integers(0, 3, 10).astype(float),
            "AMT_REQ_CREDIT_BUREAU_HOUR": rng.integers(0, 2, 10).astype(float),
            "AMT_REQ_CREDIT_BUREAU_DAY": rng.integers(0, 2, 10).astype(float),
            "AMT_REQ_CREDIT_BUREAU_WEEK": rng.integers(0, 3, 10).astype(float),
            "AMT_REQ_CREDIT_BUREAU_MON": rng.integers(0, 5, 10).astype(float),
            "AMT_REQ_CREDIT_BUREAU_QRT": rng.integers(0, 5, 10).astype(float),
            "AMT_REQ_CREDIT_BUREAU_YEAR": rng.integers(0, 10, 10).astype(float),
            "DAYS_EMPLOYED_ANOM": rng.integers(0, 2, 10).astype("int8"),
        }
    )
    result = _engineer_application_features(synth_df)
    for col in ["AMT_INCOME_TOTAL_CAPPED"] + ENGINEERED_APP_FEATURE_COLS:
        assert col in result.columns
    assert result.shape[0] == 10
    assert result["AGE_YEARS"].notna().all()
    assert result["CREDIT_INCOME_RATIO"].isna().sum() == 0
    print("  All 13 feature columns present ... PASSED")
    print("  Row count preserved .............. PASSED")
    print("  AGE_YEARS no NaN ................. PASSED")
    print("  CREDIT_INCOME_RATIO no NaN ....... PASSED")
    print("All Stage 2 tests PASSED")

    print()
    print("=" * 60)
    print("Stage 3 - Child-Table Aggregate Smoke Tests")
    print("=" * 60)
    agg_specs = [
        ("bureau", _agg_bureau, ["SK_ID_CURR"] + BUREAU_AGG_COLS, "bureau_agg"),
        ("previous", _agg_previous, ["SK_ID_CURR"] + PREVIOUS_AGG_COLS, "previous_app_agg"),
        ("installments", _agg_installments, ["SK_ID_CURR"] + INSTALLMENTS_AGG_COLS, "installments_agg"),
        ("pos_cash", _agg_pos_cash, ["SK_ID_CURR"] + POS_CASH_AGG_COLS, "pos_cash_agg"),
        ("credit_card", _agg_credit_card, ["SK_ID_CURR"] + CREDIT_CARD_AGG_COLS, "credit_card_agg"),
    ]
    try:
        for name, fn, expected_cols, agg_name in agg_specs:
            agg_result = fn("data/raw/")
            assert isinstance(agg_result, pd.DataFrame)
            assert "SK_ID_CURR" in agg_result.columns
            assert agg_result.shape[0] > 0
            assert list(agg_result.columns) == expected_cols
            assert agg_result.shape[1] == len(expected_cols)
            assert_unique_key(agg_result, "SK_ID_CURR", agg_name)
            print(f"  _agg_{name}: {agg_result.shape[0]} rows, {agg_result.shape[1]} cols")
        print("All Stage 3 smoke tests PASSED")
    except FileNotFoundError:
        print("SKIP: data/raw/ not available - run with real data")

    print()
    print("=" * 60)
    print("Stage 4 - Fit/Transform Builder Tests")
    print("=" * 60)
    train_path = _processed_split_path("train")
    if not os.path.exists(train_path):
        print("BLOCKED: Stage 4 final acceptance requires data/processed/train.pkl from Module 1")
    else:
        train_df = pd.read_pickle(train_path)
        reduced_builder = fit_reduced_builder(train_df, save_path=REDUCED_FEATURE_BUILDER_ARTIFACT_PATH)
        full_builder = fit_full_builder(train_df, raw_dir="data/raw/", save_path=FULL_FEATURE_BUILDER_ARTIFACT_PATH)
        assert os.path.exists(REDUCED_FEATURE_BUILDER_ARTIFACT_PATH)
        assert os.path.exists(FULL_FEATURE_BUILDER_ARTIFACT_PATH)
        assert os.path.exists(REDUCED_FEATURE_BUILDER_MANIFEST_PATH)
        assert os.path.exists(FULL_FEATURE_BUILDER_MANIFEST_PATH)
        from src.builder_artifacts import load_validated_builders
        loaded_builders = load_validated_builders(
            artifact_dir="artifacts",
            processed_dir=os.environ.get("DATA_PROCESSED_DIR", "data/processed/"),
            strict_artifacts=False,
        )
        assert isinstance(loaded_builders["REDUCED"], FrozenFeatureBuilder)
        assert isinstance(loaded_builders["FULL"], FrozenFeatureBuilder)
        print("  fit builders persisted and reloaded ... PASSED")
        missing_splits = [n for n in ["val_model", "val_policy", "test"] if not os.path.exists(_processed_split_path(n))]
        if missing_splits:
            print(f"BLOCKED: Full Stage 4 transform validation requires processed splits: {missing_splits}")
        else:
            split_frames = {n: pd.read_pickle(_processed_split_path(n)) for n in PROCESSED_SPLIT_NAMES}
            for split_name, split_df in split_frames.items():
                red = build_reduced(split_df, reduced_builder)
                full = build_full(split_df, full_builder, raw_dir="data/raw/")
                assert list(red.columns) == reduced_builder.encoded_columns_
                assert list(full.columns) == full_builder.encoded_columns_
                _assert_blocked_output_columns(red)
                _assert_blocked_output_columns(full)
                print(f"  exact column contract: {split_name} ... PASSED")
            red_base = build_reduced(train_df.head(64), reduced_builder, False)
            red_scaled = build_reduced(train_df.head(64), reduced_builder, True)
            changed_numeric = [
                c
                for c in reduced_builder.numeric_scale_columns_
                if not np.allclose(red_base[c].fillna(0.0), red_scaled[c].fillna(0.0))
            ]
            assert changed_numeric
            dummy_cols = [c for c in reduced_builder.encoded_columns_ if any(c.startswith(f"{cat}_") for cat in CATEGORICAL_MODEL_COLS)]
            for col in dummy_cols:
                assert np.allclose(red_base[col], red_scaled[col])
            flag_cols = [c for c in reduced_builder.encoded_columns_ if c.endswith("_IS_MISSING")]
            for col in flag_cols:
                assert np.allclose(red_base[col], red_scaled[col])
                assert set(np.unique(red_scaled[col])) <= {0.0, 1.0}
            print("  scaling contract ................. PASSED")
            flat_input = train_df.head(1)[APPLICATION_REQUIRED_INPUT_COLS].copy()
            pre_full = _build_pre_model_frame(train_df.head(1), "FULL", raw_dir="data/raw/", allow_flattened_full_input=False)
            for col in ALL_AGGREGATE_FEATURE_COLS:
                flat_input[col] = pre_full.iloc[0][col]
            runtime_full = build_full(flat_input, full_builder, raw_dir=None)
            assert list(runtime_full.columns) == full_builder.encoded_columns_
            print("  FULL flattened runtime path ...... PASSED")
