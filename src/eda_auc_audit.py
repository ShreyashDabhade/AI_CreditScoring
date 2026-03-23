"""Stable-AUC EDA audit and bundle evaluation."""

from __future__ import annotations

import json
import os
import pickle
import sys
from dataclasses import dataclass
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from configs.config import RANDOM_STATE
from src.feature_engineering import APPLICATION_REQUIRED_INPUT_COLS

EDA_PLOTS_DIR = os.environ.get("EDA_PLOTS_DIR", "notebooks/eda_plots/")
EDA_DATA_DIR = os.environ.get("EDA_DATA_DIR", "data/")
PROCESSED_DIR = os.environ.get("DATA_PROCESSED_DIR", "data/processed/")
RAW_DIR = os.environ.get("DATA_RAW_DIR", "data/raw/")
TOP_K = 15
MIN_CATEGORY_SUPPORT = 500
MIN_MISSING_RATE = 0.05
DEFAULT_AUC_IMPROVEMENT = 0.002
DEFAULT_GAP_TOLERANCE = 0.005
RUNTIME_BASELINE_COLUMNS = tuple(APPLICATION_REQUIRED_INPUT_COLS)
HOUSING_COLUMNS = (
    "APARTMENTS_AVG",
    "BASEMENTAREA_AVG",
    "YEARS_BEGINEXPLUATATION_AVG",
    "ELEVATORS_AVG",
    "ENTRANCES_AVG",
    "FLOORSMAX_AVG",
    "LIVINGAREA_AVG",
    "NONLIVINGAREA_AVG",
    "TOTALAREA_MODE",
    "FONDKAPREMONT_MODE",
    "HOUSETYPE_MODE",
    "WALLSMATERIAL_MODE",
    "EMERGENCYSTATE_MODE",
)
RAW_PHASE2_REQUIRED_TABLES = (
    "bureau.csv",
    "previous_application.csv",
    "installments_payments.csv",
    "POS_CASH_balance.csv",
    "credit_card_balance.csv",
)


@dataclass(frozen=True)
class BundleSpec:
    name: str
    description: str
    add_columns: tuple[str, ...] = ()
    drop_columns: tuple[str, ...] = ()
    deployment_eligible: bool = True
    phase: str = "processed"


@dataclass(frozen=True)
class BundleResult:
    name: str
    description: str
    deployment_eligible: bool
    phase: str
    feature_count: int
    train_auc: float
    val_model_auc: float
    val_policy_auc: float
    test_auc: float
    train_test_gap: float
    val_policy_delta: float
    test_delta: float
    gap_increase: float
    adv_auc: float
    adv_auc_delta: float
    accepted: bool
    top_drift_features: tuple[str, ...]


def _load_pickle_df(path: str) -> pd.DataFrame:
    with open(path, "rb") as handle:
        df = pickle.load(handle)
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{path} did not contain a pandas DataFrame")
    return df


def load_processed_splits(processed_dir: str = PROCESSED_DIR) -> dict[str, pd.DataFrame]:
    return {
        "train": _load_pickle_df(os.path.join(processed_dir, "train.pkl")),
        "val_model": _load_pickle_df(os.path.join(processed_dir, "val_model.pkl")),
        "val_policy": _load_pickle_df(os.path.join(processed_dir, "val_policy.pkl")),
        "test": _load_pickle_df(os.path.join(processed_dir, "test.pkl")),
    }


def load_adversarial_dataset(processed_dir: str = PROCESSED_DIR) -> dict[str, pd.DataFrame]:
    path = os.path.join(processed_dir, "app_test_adv.pkl")
    with open(path, "rb") as handle:
        data = pickle.load(handle)
    if not isinstance(data, dict) or set(data) != {"adv_train", "adv_val"}:
        raise TypeError(f"{path} must contain adv_train and adv_val")
    return data


def _safe_div(a: pd.Series | np.ndarray, b: pd.Series | np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where((pd.notna(a)) & (pd.notna(b)) & (b != 0), a / b, np.nan)


def engineer_candidate_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "AMT_INCOME_TOTAL_CAPPED" not in out.columns and "AMT_INCOME_TOTAL" in out.columns:
        out["AMT_INCOME_TOTAL_CAPPED"] = out["AMT_INCOME_TOTAL"]
    if "DAYS_EMPLOYED_ANOM" not in out.columns and "DAYS_EMPLOYED" in out.columns:
        out["DAYS_EMPLOYED_ANOM"] = (out["DAYS_EMPLOYED"] == 365243).astype("int8")
        out["DAYS_EMPLOYED"] = out["DAYS_EMPLOYED"].replace(365243, np.nan)
    out["AGE_YEARS"] = -out["DAYS_BIRTH"] / 365
    out["EMPLOYMENT_YEARS"] = -out["DAYS_EMPLOYED"] / 365
    out["EXT_SOURCE_MEAN"] = out[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].mean(axis=1)
    out["EXT_SOURCE_STD"] = out[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].std(axis=1)
    out["EXT_SOURCE_COUNT"] = out[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].notna().sum(axis=1)
    out["CREDIT_INCOME_RATIO"] = _safe_div(out["AMT_CREDIT"], out["AMT_INCOME_TOTAL_CAPPED"])
    out["ANNUITY_INCOME_RATIO"] = _safe_div(out["AMT_ANNUITY"], out["AMT_INCOME_TOTAL_CAPPED"])
    out["INCOME_PER_FAM_MEMBER"] = _safe_div(out["AMT_INCOME_TOTAL_CAPPED"], out["CNT_FAM_MEMBERS"])
    out["CREDIT_PER_FAM_MEMBER"] = _safe_div(out["AMT_CREDIT"], out["CNT_FAM_MEMBERS"])
    out["EXT_SOURCE_AGE_INTERACTION"] = out["EXT_SOURCE_MEAN"] * out["AGE_YEARS"]
    out["EXT_SOURCE_EMPLOYMENT_INTERACTION"] = out["EXT_SOURCE_MEAN"] * out["EMPLOYMENT_YEARS"]
    out["EXT_SOURCE_BURDEN_RATIO"] = _safe_div(out["EXT_SOURCE_MEAN"], out["ANNUITY_INCOME_RATIO"])
    out["EXT_SOURCE_CREDIT_RATIO"] = _safe_div(out["EXT_SOURCE_MEAN"], out["CREDIT_INCOME_RATIO"])
    housing_numeric = [c for c in HOUSING_COLUMNS if c in out.columns and pd.api.types.is_numeric_dtype(out[c])]
    housing_all = [c for c in HOUSING_COLUMNS if c in out.columns]
    out["HOUSING_MISSING_COUNT"] = out[housing_all].isna().sum(axis=1) if housing_all else np.nan
    out["HOUSING_QUALITY_MEAN"] = out[housing_numeric].mean(axis=1) if housing_numeric else np.nan
    out["HOUSING_QUALITY_STD"] = out[housing_numeric].std(axis=1) if housing_numeric else np.nan
    return out


def _bundle_specs() -> list[BundleSpec]:
    return [
        BundleSpec(
            name="stable_application_bundle",
            description="Stable EXT_SOURCE, age/employment, and family-normalized affordability features",
            add_columns=(
                "AGE_YEARS",
                "EMPLOYMENT_YEARS",
                "EXT_SOURCE_MEAN",
                "EXT_SOURCE_STD",
                "EXT_SOURCE_COUNT",
                "EXT_SOURCE_AGE_INTERACTION",
                "EXT_SOURCE_BURDEN_RATIO",
                "CREDIT_INCOME_RATIO",
                "ANNUITY_INCOME_RATIO",
                "INCOME_PER_FAM_MEMBER",
                "CREDIT_PER_FAM_MEMBER",
            ),
            deployment_eligible=True,
            phase="processed",
        ),
        BundleSpec(
            name="housing_bundle",
            description="Compressed housing completeness and quality summaries from raw housing columns",
            add_columns=("HOUSING_MISSING_COUNT", "HOUSING_QUALITY_MEAN", "HOUSING_QUALITY_STD"),
            deployment_eligible=False,
            phase="processed",
        ),
        BundleSpec(
            name="drift_control_bundle",
            description="Exclude the most drift-prone runtime feature from the model matrix",
            drop_columns=("DAYS_EMPLOYED_ANOM",),
            deployment_eligible=True,
            phase="processed",
        ),
        BundleSpec(
            name="raw_table_bundle",
            description="Recent-window bureau and repayment features from raw child tables",
            deployment_eligible=True,
            phase="raw",
        ),
    ]


def _filter_feature_columns(
    df: pd.DataFrame,
    base_columns: tuple[str, ...],
    bundle: BundleSpec | None = None,
    raw_phase_columns: tuple[str, ...] = (),
) -> list[str]:
    columns = [c for c in base_columns if c in df.columns]
    if bundle is not None:
        columns.extend([c for c in bundle.add_columns if c in df.columns])
        if bundle.phase == "raw":
            columns.extend([c for c in raw_phase_columns if c in df.columns])
        columns = [c for c in columns if c not in set(bundle.drop_columns)]
    return list(dict.fromkeys(columns))


def _prepare_model_frames(
    train_df: pd.DataFrame,
    others: dict[str, pd.DataFrame],
    feature_columns: list[str],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    train = train_df[feature_columns].copy()
    transformed_others = {name: frame[feature_columns].copy() for name, frame in others.items()}
    numeric_cols = train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in train.columns if c not in numeric_cols]

    medians = {
        c: float(train[c].median(skipna=True)) if train[c].notna().any() else 0.0
        for c in numeric_cols
    }
    for col, value in medians.items():
        train[col] = train[col].replace([np.inf, -np.inf], np.nan).fillna(value)
        for frame in transformed_others.values():
            frame[col] = frame[col].replace([np.inf, -np.inf], np.nan).fillna(value)

    for col in categorical_cols:
        train[col] = train[col].fillna("MISSING").astype(str)
        for frame in transformed_others.values():
            frame[col] = frame[col].fillna("MISSING").astype(str)

    encoded_train = pd.get_dummies(train, columns=categorical_cols, dtype=float, drop_first=False)
    encoded_others: dict[str, pd.DataFrame] = {}
    for name, frame in transformed_others.items():
        encoded = pd.get_dummies(frame, columns=categorical_cols, dtype=float, drop_first=False)
        encoded_others[name] = encoded.reindex(columns=encoded_train.columns, fill_value=0.0)
    return encoded_train, encoded_others


def _fit_auc_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> XGBClassifier:
    positives = float(y_train.sum())
    negatives = float(len(y_train) - positives)
    scale_pos_weight = negatives / positives if positives > 0 else 1.0
    model = XGBClassifier(
        n_estimators=250,
        learning_rate=0.05,
        max_depth=4,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="auc",
        random_state=RANDOM_STATE,
        tree_method="hist",
        early_stopping_rounds=30,
        scale_pos_weight=scale_pos_weight,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


def _score_auc(model: XGBClassifier, X: pd.DataFrame, y: pd.Series) -> float:
    return float(roc_auc_score(y, model.predict_proba(X)[:, 1]))


def evaluate_bundle_model(
    audit_splits: dict[str, pd.DataFrame],
    bundle: BundleSpec | None,
    raw_phase_columns: tuple[str, ...] = (),
) -> dict[str, Any]:
    feature_columns = _filter_feature_columns(
        audit_splits["train"],
        RUNTIME_BASELINE_COLUMNS,
        bundle=bundle,
        raw_phase_columns=raw_phase_columns,
    )
    X_train, encoded_others = _prepare_model_frames(
        audit_splits["train"],
        {
            "val_model": audit_splits["val_model"],
            "val_policy": audit_splits["val_policy"],
            "test": audit_splits["test"],
        },
        feature_columns,
    )
    y_train = audit_splits["train"]["TARGET"].astype(int)
    y_val_model = audit_splits["val_model"]["TARGET"].astype(int)
    y_val_policy = audit_splits["val_policy"]["TARGET"].astype(int)
    y_test = audit_splits["test"]["TARGET"].astype(int)
    model = _fit_auc_model(X_train, y_train, encoded_others["val_model"], y_val_model)
    return {
        "model": model,
        "feature_columns": feature_columns,
        "encoded_feature_columns": tuple(X_train.columns),
        "train_auc": _score_auc(model, X_train, y_train),
        "val_model_auc": _score_auc(model, encoded_others["val_model"], y_val_model),
        "val_policy_auc": _score_auc(model, encoded_others["val_policy"], y_val_policy),
        "test_auc": _score_auc(model, encoded_others["test"], y_test),
    }


def evaluate_adversarial_drift(
    adv_data: dict[str, pd.DataFrame],
    bundle: BundleSpec | None,
    raw_phase_columns: tuple[str, ...] = (),
) -> dict[str, Any]:
    adv_train = engineer_candidate_features(adv_data["adv_train"])
    adv_val = engineer_candidate_features(adv_data["adv_val"])
    feature_columns = _filter_feature_columns(
        adv_train,
        RUNTIME_BASELINE_COLUMNS,
        bundle=bundle,
        raw_phase_columns=raw_phase_columns,
    )
    X_train, encoded_others = _prepare_model_frames(adv_train, {"adv_val": adv_val}, feature_columns)
    y_train = adv_train["ADV_LABEL"].astype(int)
    y_val = adv_val["ADV_LABEL"].astype(int)
    if len(X_train) > 60_000:
        sample_idx = np.random.default_rng(RANDOM_STATE).choice(len(X_train), size=60_000, replace=False)
        X_train = X_train.iloc[sample_idx]
        y_train = y_train.iloc[sample_idx]
    model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.08,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="auc",
        random_state=RANDOM_STATE,
        tree_method="hist",
    )
    model.fit(X_train, y_train, eval_set=[(encoded_others["adv_val"], y_val)], verbose=False)
    probs = model.predict_proba(encoded_others["adv_val"])[:, 1]
    importance = pd.Series(model.feature_importances_, index=X_train.columns, dtype=float).sort_values(ascending=False)
    return {"adv_auc": float(roc_auc_score(y_val, probs)), "importance": importance}


def _numeric_auc_summary(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [c for c in train_df.select_dtypes(include=[np.number]).columns if c != "TARGET"]
    rows: list[dict[str, Any]] = []
    for col in numeric_cols:
        tr = train_df[[col, "TARGET"]].replace([np.inf, -np.inf], np.nan).dropna()
        te = test_df[[col, "TARGET"]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(tr) < 1000 or len(te) < 1000 or tr[col].nunique() < 2 or te[col].nunique() < 2:
            continue
        train_auc_raw = float(roc_auc_score(tr["TARGET"], tr[col]))
        test_auc_raw = float(roc_auc_score(te["TARGET"], te[col]))
        train_auc = max(train_auc_raw, 1 - train_auc_raw)
        test_auc = max(test_auc_raw, 1 - test_auc_raw)
        rows.append(
            {
                "feature": col,
                "train_auc": train_auc,
                "test_auc": test_auc,
                "auc_drop": train_auc - test_auc,
                "missing_rate_train": float(train_df[col].isna().mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["test_auc", "auc_drop"], ascending=[False, True])


def _categorical_lift_summary(train_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for col in train_df.select_dtypes(exclude=[np.number]).columns:
        stats = train_df.groupby(col, dropna=False)["TARGET"].agg(["mean", "size"]).reset_index()
        stats = stats[stats["size"] >= MIN_CATEGORY_SUPPORT]
        if len(stats) < 2:
            continue
        high = stats.loc[stats["mean"].idxmax()]
        low = stats.loc[stats["mean"].idxmin()]
        rows.append(
            {
                "feature": col,
                "target_rate_gap": float(high["mean"] - low["mean"]),
                "highest_category": str(high[col]),
                "highest_category_rate": float(high["mean"]),
                "highest_category_n": int(high["size"]),
                "lowest_category": str(low[col]),
                "lowest_category_rate": float(low["mean"]),
                "lowest_category_n": int(low["size"]),
            }
        )
    return pd.DataFrame(rows).sort_values("target_rate_gap", ascending=False)


def _missingness_lift_summary(train_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for col, rate in train_df.isna().mean().sort_values(ascending=False).items():
        if rate < MIN_MISSING_RATE:
            continue
        mask = train_df[col].isna()
        if mask.all() or (~mask).all():
            continue
        rows.append(
            {
                "feature": col,
                "missing_rate": float(rate),
                "missing_target_rate": float(train_df.loc[mask, "TARGET"].mean()),
                "present_target_rate": float(train_df.loc[~mask, "TARGET"].mean()),
                "target_gap": float(train_df.loc[mask, "TARGET"].mean() - train_df.loc[~mask, "TARGET"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("target_gap", ascending=False)


def _split_drift_summary(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for col in [c for c in train_df.select_dtypes(include=[np.number]).columns if c != "TARGET"]:
        train_mean = float(train_df[col].mean(skipna=True))
        test_mean = float(test_df[col].mean(skipna=True))
        train_std = float(train_df[col].std(skipna=True))
        denom = train_std if np.isfinite(train_std) and train_std > 1e-9 else 1.0
        rows.append(
            {
                "feature": col,
                "mean_shift_z": abs(test_mean - train_mean) / denom if np.isfinite(train_mean) and np.isfinite(test_mean) else np.nan,
                "missing_rate_train": float(train_df[col].isna().mean()),
                "missing_rate_test": float(test_df[col].isna().mean()),
                "missing_rate_gap": float(test_df[col].isna().mean() - train_df[col].isna().mean()),
                "train_mean": train_mean,
                "test_mean": test_mean,
            }
        )
    return pd.DataFrame(rows).sort_values("mean_shift_z", ascending=False)


def _candidate_interaction_summary(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    features = [
        "EXT_SOURCE_MEAN",
        "EXT_SOURCE_STD",
        "EXT_SOURCE_COUNT",
        "AGE_YEARS",
        "EMPLOYMENT_YEARS",
        "CREDIT_INCOME_RATIO",
        "ANNUITY_INCOME_RATIO",
        "INCOME_PER_FAM_MEMBER",
        "CREDIT_PER_FAM_MEMBER",
        "EXT_SOURCE_AGE_INTERACTION",
        "EXT_SOURCE_EMPLOYMENT_INTERACTION",
        "EXT_SOURCE_BURDEN_RATIO",
        "EXT_SOURCE_CREDIT_RATIO",
        "HOUSING_MISSING_COUNT",
        "HOUSING_QUALITY_MEAN",
    ]
    rows: list[dict[str, Any]] = []
    for col in features:
        if col not in train_df.columns or col not in test_df.columns:
            continue
        tr = train_df[[col, "TARGET"]].replace([np.inf, -np.inf], np.nan).dropna()
        te = test_df[[col, "TARGET"]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(tr) < 1000 or len(te) < 1000 or tr[col].nunique() < 2 or te[col].nunique() < 2:
            continue
        train_auc_raw = float(roc_auc_score(tr["TARGET"], tr[col]))
        test_auc_raw = float(roc_auc_score(te["TARGET"], te[col]))
        train_auc = max(train_auc_raw, 1 - train_auc_raw)
        test_auc = max(test_auc_raw, 1 - test_auc_raw)
        rows.append(
            {
                "feature": col,
                "train_auc": train_auc,
                "test_auc": test_auc,
                "auc_drop": train_auc - test_auc,
                "missing_rate_train": float(train_df[col].isna().mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["test_auc", "auc_drop"], ascending=[False, True])


def _save_dataframe(df: pd.DataFrame, filename: str) -> str:
    os.makedirs(EDA_DATA_DIR, exist_ok=True)
    path = os.path.join(EDA_DATA_DIR, filename)
    df.to_csv(path, index=False)
    return path


def _plot_bar(df: pd.DataFrame, x: str, y: str, title: str, path: str, color: str) -> str:
    os.makedirs(EDA_PLOTS_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=df, x=x, y=y, color=color, ax=ax)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=60)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def _plot_split_target_rates(split_rates: pd.DataFrame) -> str:
    path = os.path.join(EDA_PLOTS_DIR, "stable_auc_split_target_rate_drift.png")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(
        data=split_rates,
        x="split",
        y="default_rate",
        hue="split",
        palette=["#1D9E75", "#378ADD", "#D85A30", "#7F77DD"],
        legend=False,
        ax=ax,
    )
    ax.set_title("Default rate drift across proxy-time splits")
    ax.set_ylabel("Default rate")
    for idx, row in split_rates.iterrows():
        ax.text(idx, row["default_rate"], f"{row['default_rate']:.3f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def _plot_numeric_auc_stability(df: pd.DataFrame) -> str:
    path = os.path.join(EDA_PLOTS_DIR, "stable_auc_numeric_auc_stability_top15.png")
    top = df.head(TOP_K).copy()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top["feature"], top["test_auc"], color="#1D9E75")
    for idx, row in top.reset_index(drop=True).iterrows():
        ax.text(row["test_auc"], idx, f" drop {row['auc_drop']:+.3f}", va="center")
    ax.set_title("Top stable numeric signals by test AUC")
    ax.set_xlabel("Test univariate ROC-AUC")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def _plot_bundle_comparison(df: pd.DataFrame) -> str:
    path = os.path.join(EDA_PLOTS_DIR, "stable_auc_bundle_comparison.png")
    plot_df = df[["name", "val_policy_auc", "test_auc"]].melt(id_vars="name", var_name="split", value_name="auc")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=plot_df, x="name", y="auc", hue="split", ax=ax)
    ax.set_title("Baseline vs bundle comparison")
    ax.tick_params(axis="x", rotation=40)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def _raw_phase2_availability(raw_dir: str = RAW_DIR) -> dict[str, Any]:
    missing = [name for name in RAW_PHASE2_REQUIRED_TABLES if not os.path.exists(os.path.join(raw_dir, name))]
    if missing:
        return {
            "status": "skipped",
            "missing_tables": missing,
            "message": "Raw child tables are not available locally yet; phase 2 recency-window audit is deferred.",
        }
    return {
        "status": "ready",
        "missing_tables": [],
        "message": "Raw child tables are available; phase 2 recency-window audit can run.",
    }


def _feature_bucket_summary(
    numeric_auc: pd.DataFrame,
    split_drift: pd.DataFrame,
    adversarial_top_features: set[str],
) -> pd.DataFrame:
    merged = numeric_auc.merge(split_drift[["feature", "mean_shift_z"]], on="feature", how="left")
    rows: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        feature = row["feature"]
        if feature == "DAYS_EMPLOYED_ANOM" or feature in adversarial_top_features or row["mean_shift_z"] >= 0.5:
            bucket = "avoid_or_cap"
        elif row["test_auc"] >= 0.56 and row["auc_drop"] <= 0.015 and row["mean_shift_z"] < 0.35:
            bucket = "keep_prioritize"
        else:
            bucket = "use_carefully"
        rows.append(
            {
                "feature": feature,
                "bucket": bucket,
                "test_auc": row["test_auc"],
                "auc_drop": row["auc_drop"],
                "mean_shift_z": row["mean_shift_z"],
                "is_top_adversarial_driver": feature in adversarial_top_features,
            }
        )
    return pd.DataFrame(rows).sort_values(["bucket", "test_auc"], ascending=[True, False])


def _top_findings_table(
    split_rates: pd.DataFrame,
    missingness: pd.DataFrame,
    numeric_auc: pd.DataFrame,
    categorical_lift: pd.DataFrame,
    adversarial_importance: pd.Series,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for idx, row in split_rates.iterrows():
        rows.append({"section": "split_target_rates", "rank": idx + 1, "feature": row["split"], "metric": "default_rate", "value": row["default_rate"], "detail": ""})
    for idx, row in missingness.head(5).reset_index(drop=True).iterrows():
        rows.append({"section": "missingness_lift", "rank": idx + 1, "feature": row["feature"], "metric": "target_gap", "value": row["target_gap"], "detail": f"missing_rate={row['missing_rate']:.3f}"})
    for idx, row in numeric_auc.head(5).reset_index(drop=True).iterrows():
        rows.append({"section": "stable_numeric_auc", "rank": idx + 1, "feature": row["feature"], "metric": "test_auc", "value": row["test_auc"], "detail": f"auc_drop={row['auc_drop']:+.3f}"})
    for idx, row in categorical_lift.head(5).reset_index(drop=True).iterrows():
        rows.append({"section": "categorical_lift", "rank": idx + 1, "feature": row["feature"], "metric": "target_rate_gap", "value": row["target_rate_gap"], "detail": f"{row['highest_category']} vs {row['lowest_category']}"})
    for idx, (feature, importance) in enumerate(adversarial_importance.head(5).items(), start=1):
        rows.append({"section": "adversarial_drift", "rank": idx, "feature": feature, "metric": "importance", "value": float(importance), "detail": ""})
    return pd.DataFrame(rows)


def run_stable_auc_audit(
    *,
    processed_dir: str = PROCESSED_DIR,
    raw_dir: str = RAW_DIR,
) -> dict[str, Any]:
    os.makedirs(EDA_DATA_DIR, exist_ok=True)
    os.makedirs(EDA_PLOTS_DIR, exist_ok=True)

    splits = {name: engineer_candidate_features(df) for name, df in load_processed_splits(processed_dir).items()}
    adv_data = load_adversarial_dataset(processed_dir)

    split_rates = pd.DataFrame(
        {
            "split": list(splits.keys()),
            "default_rate": [float(frame["TARGET"].mean()) for frame in splits.values()],
            "rows": [int(len(frame)) for frame in splits.values()],
        }
    )
    missingness = _missingness_lift_summary(splits["train"])
    numeric_auc = _numeric_auc_summary(splits["train"], splits["test"])
    categorical_lift = _categorical_lift_summary(splits["train"])
    split_drift = _split_drift_summary(splits["train"], splits["test"])
    candidate_interactions = _candidate_interaction_summary(splits["train"], splits["test"])

    baseline_bundle = BundleSpec(name="baseline", description="Runtime-available raw application inputs only")
    baseline_model = evaluate_bundle_model(splits, baseline_bundle)
    baseline_adv = evaluate_adversarial_drift(adv_data, baseline_bundle)

    bundle_results: list[BundleResult] = [
        BundleResult(
            name="baseline",
            description=baseline_bundle.description,
            deployment_eligible=True,
            phase="processed",
            feature_count=len(baseline_model["feature_columns"]),
            train_auc=baseline_model["train_auc"],
            val_model_auc=baseline_model["val_model_auc"],
            val_policy_auc=baseline_model["val_policy_auc"],
            test_auc=baseline_model["test_auc"],
            train_test_gap=baseline_model["train_auc"] - baseline_model["test_auc"],
            val_policy_delta=0.0,
            test_delta=0.0,
            gap_increase=0.0,
            adv_auc=baseline_adv["adv_auc"],
            adv_auc_delta=0.0,
            accepted=False,
            top_drift_features=tuple(baseline_adv["importance"].head(5).index.tolist()),
        )
    ]

    raw_phase = _raw_phase2_availability(raw_dir)
    raw_phase_columns: tuple[str, ...] = ()
    for bundle in _bundle_specs():
        if bundle.phase == "raw" and raw_phase["status"] != "ready":
            bundle_results.append(
                BundleResult(
                    name=bundle.name,
                    description=bundle.description,
                    deployment_eligible=bundle.deployment_eligible,
                    phase=bundle.phase,
                    feature_count=len(RUNTIME_BASELINE_COLUMNS),
                    train_auc=np.nan,
                    val_model_auc=np.nan,
                    val_policy_auc=np.nan,
                    test_auc=np.nan,
                    train_test_gap=np.nan,
                    val_policy_delta=np.nan,
                    test_delta=np.nan,
                    gap_increase=np.nan,
                    adv_auc=np.nan,
                    adv_auc_delta=np.nan,
                    accepted=False,
                    top_drift_features=(),
                )
            )
            continue

        model_eval = evaluate_bundle_model(splits, bundle, raw_phase_columns=raw_phase_columns)
        adv_eval = evaluate_adversarial_drift(adv_data, bundle, raw_phase_columns=raw_phase_columns)
        train_test_gap = model_eval["train_auc"] - model_eval["test_auc"]
        val_policy_delta = model_eval["val_policy_auc"] - baseline_model["val_policy_auc"]
        test_delta = model_eval["test_auc"] - baseline_model["test_auc"]
        gap_increase = train_test_gap - (baseline_model["train_auc"] - baseline_model["test_auc"])
        top_drift_features = tuple(adv_eval["importance"].head(5).index.tolist())
        introduces_top_drift_feature = any(feature in top_drift_features for feature in bundle.add_columns)
        accepted = (
            bundle.deployment_eligible
            and max(val_policy_delta, test_delta) >= DEFAULT_AUC_IMPROVEMENT
            and gap_increase <= DEFAULT_GAP_TOLERANCE
            and (adv_eval["adv_auc"] - baseline_adv["adv_auc"]) <= 0.005
            and not introduces_top_drift_feature
        )
        bundle_results.append(
            BundleResult(
                name=bundle.name,
                description=bundle.description,
                deployment_eligible=bundle.deployment_eligible,
                phase=bundle.phase,
                feature_count=len(model_eval["feature_columns"]),
                train_auc=model_eval["train_auc"],
                val_model_auc=model_eval["val_model_auc"],
                val_policy_auc=model_eval["val_policy_auc"],
                test_auc=model_eval["test_auc"],
                train_test_gap=train_test_gap,
                val_policy_delta=val_policy_delta,
                test_delta=test_delta,
                gap_increase=gap_increase,
                adv_auc=adv_eval["adv_auc"],
                adv_auc_delta=adv_eval["adv_auc"] - baseline_adv["adv_auc"],
                accepted=accepted,
                top_drift_features=top_drift_features,
            )
        )

    bundle_df = pd.DataFrame(
        [result.__dict__ | {"top_drift_features": "|".join(result.top_drift_features)} for result in bundle_results]
    )
    adversarial_df = baseline_adv["importance"].head(50).rename_axis("feature").reset_index(name="importance")
    adversarial_top_features = set(adversarial_df.head(10)["feature"].tolist())
    feature_buckets = _feature_bucket_summary(numeric_auc, split_drift, adversarial_top_features)
    top_findings = _top_findings_table(split_rates, missingness, numeric_auc, categorical_lift, baseline_adv["importance"])

    _save_dataframe(split_rates, "stable_auc_split_target_rates.csv")
    _save_dataframe(missingness, "stable_auc_missingness_lift.csv")
    _save_dataframe(numeric_auc, "stable_auc_numeric_univariate_auc.csv")
    _save_dataframe(categorical_lift, "stable_auc_categorical_target_lift.csv")
    _save_dataframe(split_drift, "stable_auc_feature_drift_summary.csv")
    _save_dataframe(candidate_interactions, "stable_auc_candidate_interaction_scan.csv")
    _save_dataframe(adversarial_df, "stable_auc_adversarial_drift_importance.csv")
    _save_dataframe(bundle_df, "stable_auc_bundle_evaluation.csv")
    _save_dataframe(feature_buckets, "stable_auc_feature_buckets.csv")
    _save_dataframe(top_findings, "stable_auc_top_findings.csv")

    plot_paths = {
        "split_target_rates": _plot_split_target_rates(split_rates),
        "missingness_lift": _plot_bar(missingness.head(TOP_K), "feature", "target_gap", "Missingness target lift - top features", os.path.join(EDA_PLOTS_DIR, "stable_auc_missingness_target_lift_top15.png"), "#D85A30"),
        "numeric_auc_stability": _plot_numeric_auc_stability(numeric_auc),
        "categorical_lift": _plot_bar(categorical_lift.head(TOP_K), "feature", "target_rate_gap", "Categorical target-rate separation - top features", os.path.join(EDA_PLOTS_DIR, "stable_auc_categorical_lift_top15.png"), "#378ADD"),
        "adversarial_drift": _plot_bar(adversarial_df.head(TOP_K), "feature", "importance", "Top adversarial drift drivers", os.path.join(EDA_PLOTS_DIR, "stable_auc_adversarial_drift_top15.png"), "#7F77DD"),
        "candidate_interactions": _plot_bar(candidate_interactions.head(TOP_K), "feature", "test_auc", "Candidate interaction stability scan", os.path.join(EDA_PLOTS_DIR, "stable_auc_candidate_interaction_scan.png"), "#1D9E75"),
        "bundle_comparison": _plot_bundle_comparison(bundle_df[bundle_df["phase"] == "processed"]),
    }

    summary = {
        "scope": "hybrid",
        "optimization_goal": "stable_auc",
        "baseline": {
            "train_auc": baseline_model["train_auc"],
            "val_model_auc": baseline_model["val_model_auc"],
            "val_policy_auc": baseline_model["val_policy_auc"],
            "test_auc": baseline_model["test_auc"],
            "adversarial_auc": baseline_adv["adv_auc"],
        },
        "accepted_deployment_bundles": bundle_df.loc[bundle_df["accepted"] == True, "name"].tolist(),  # noqa: E712
        "rejected_or_deferred_bundles": bundle_df.loc[bundle_df["accepted"] != True, "name"].tolist(),  # noqa: E712
        "top_drift_drivers": adversarial_df.head(10)["feature"].tolist(),
        "top_stable_numeric_features": numeric_auc.head(10)["feature"].tolist(),
        "top_categorical_separators": categorical_lift.head(10)["feature"].tolist(),
        "recommended_runtime_bundle_profile": {
            "MODEL_APPLICATION_BUNDLES": "",
            "MODEL_DRIFT_CONTROL_BUNDLES": "",
            "MODEL_FULL_BUNDLES": "",
        },
        "raw_phase2_status": raw_phase,
        "plot_paths": plot_paths,
    }
    with open(os.path.join(EDA_DATA_DIR, "stable_auc_audit_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


if __name__ == "__main__":
    print(json.dumps(run_stable_auc_audit(), indent=2))
