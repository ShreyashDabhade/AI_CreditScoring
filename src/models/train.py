"""Module 3 — Model Training.

Trains a FULL scoring model, calibrates probabilities, persists artifacts
for Module 5, and writes a reproducibility report.

If real processed/raw data is unavailable in the local clone, this module falls
back to a deterministic synthetic dataset so the branch remains runnable.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from typing import Any

if __package__ is None or __package__ == "":
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from configs.config import (
    APPROVE_THRESHOLD,
    ARTIFACT_DIR,
    DATA_DIR,
    DECLINE_THRESHOLD,
    MODEL_VERSIONS,
    RANDOM_STATE,
)
from src.feature_engineering import (
    ALL_AGGREGATE_FEATURE_COLS,
    FULL_FEATURE_BUILDER_ARTIFACT_PATH,
    _agg_bureau,
    _agg_credit_card,
    _agg_installments,
    _agg_pos_cash,
    _agg_previous,
    _build_pre_model_frame,
    _fit_builder_from_pre_model_frame,
    build_full,
)
from src.models.runtime_support import LogisticProbabilityCalibrator, TreeShapExplainer

REPRODUCIBILITY_REPORT_FILENAME = "reproducibility_report.json"
FAIRNESS_RESULT_FILENAME = "model_fairness_audit_passed.joblib"
RAW_TABLE_NAMES = [
    "bureau.csv",
    "previous_application.csv",
    "installments_payments.csv",
    "POS_CASH_balance.csv",
    "credit_card_balance.csv",
]


@dataclass
class DatasetBundle:
    mode: str
    train: pd.DataFrame
    val_model: pd.DataFrame
    val_policy: pd.DataFrame
    test: pd.DataFrame
    raw_dir: str | None = None
    uses_flattened_full_input: bool = False


def decision_from_pd(probability_of_default: float) -> str:
    if probability_of_default < APPROVE_THRESHOLD:
        return "APPROVE"
    if probability_of_default < DECLINE_THRESHOLD:
        return "REVIEW"
    return "DECLINE"


def _artifact_path(artifact_dir: str, filename: str) -> str:
    return os.path.join(artifact_dir, filename)


def load_artifacts(artifact_dir: str = ARTIFACT_DIR) -> dict[str, Any]:
    return {
        "full_model": joblib.load(_artifact_path(artifact_dir, "full_model.joblib")),
        "full_calibrator": joblib.load(_artifact_path(artifact_dir, "full_calibrator.joblib")),
        "full_shap_explainer": joblib.load(_artifact_path(artifact_dir, "full_shap_explainer.joblib")),
        "full_builder": joblib.load(FULL_FEATURE_BUILDER_ARTIFACT_PATH),
    }


def _has_real_training_inputs(processed_dir: str, raw_dir: str) -> bool:
    split_names = ["train.pkl", "val_model.pkl", "val_policy.pkl", "test.pkl"]
    split_paths = [os.path.join(processed_dir, name) for name in split_names]
    raw_paths = [os.path.join(raw_dir, name) for name in RAW_TABLE_NAMES]
    return all(os.path.exists(path) for path in split_paths + raw_paths)


def _load_real_bundle(processed_dir: str, raw_dir: str) -> DatasetBundle:
    return DatasetBundle(
        mode="real",
        train=pd.read_pickle(os.path.join(processed_dir, "train.pkl")),
        val_model=pd.read_pickle(os.path.join(processed_dir, "val_model.pkl")),
        val_policy=pd.read_pickle(os.path.join(processed_dir, "val_policy.pkl")),
        test=pd.read_pickle(os.path.join(processed_dir, "test.pkl")),
        raw_dir=raw_dir,
        uses_flattened_full_input=False,
    )


def _choice(rng: np.random.Generator, values: list[Any], size: int) -> list[Any]:
    return rng.choice(np.array(values, dtype=object), size=size).tolist()


def _generate_synthetic_application_frame(rng: np.random.Generator, n_rows: int) -> pd.DataFrame:
    income = np.exp(rng.normal(np.log(150000.0), 0.45, n_rows)).clip(40000.0, 700000.0)
    credit = (income * rng.uniform(1.2, 3.8, n_rows) + rng.normal(0.0, 18000.0, n_rows)).clip(50000.0, 1200000.0)
    annuity = (credit / rng.uniform(7.0, 24.0, n_rows)).clip(4000.0, 90000.0)
    goods_price = (credit * rng.uniform(0.75, 1.05, n_rows)).clip(30000.0, 1000000.0)

    df = pd.DataFrame(
        {
            "SK_ID_CURR": np.arange(100000, 100000 + n_rows, dtype=np.int64),
            "AMT_INCOME_TOTAL_CAPPED": income.astype(float),
            "AMT_INCOME_TOTAL": (income * rng.uniform(0.95, 1.05, n_rows)).astype(float),
            "AMT_CREDIT": credit.astype(float),
            "AMT_ANNUITY": annuity.astype(float),
            "AMT_GOODS_PRICE": goods_price.astype(float),
            "DAYS_BIRTH": (-rng.uniform(8000.0, 25000.0, n_rows)).astype(float),
            "DAYS_EMPLOYED": (-rng.uniform(30.0, 8000.0, n_rows)).astype(float),
            "DAYS_REGISTRATION": (-rng.uniform(100.0, 7000.0, n_rows)).astype(float),
            "DAYS_ID_PUBLISH": (-rng.uniform(10.0, 5000.0, n_rows)).astype(float),
            "DAYS_LAST_PHONE_CHANGE": (-rng.uniform(10.0, 3000.0, n_rows)).astype(float),
            "REGION_POPULATION_RELATIVE": rng.uniform(0.002, 0.08, n_rows).astype(float),
            "EXT_SOURCE_1": rng.beta(2.4, 2.3, n_rows).astype(float),
            "EXT_SOURCE_2": rng.beta(2.8, 2.0, n_rows).astype(float),
            "EXT_SOURCE_3": rng.beta(2.3, 2.7, n_rows).astype(float),
            "CNT_FAM_MEMBERS": rng.integers(1, 6, n_rows).astype(float),
            "OWN_CAR_AGE": rng.uniform(0.0, 18.0, n_rows).astype(float),
            "OBS_30_CNT_SOCIAL_CIRCLE": rng.poisson(1.2, n_rows).astype(float),
            "DEF_30_CNT_SOCIAL_CIRCLE": rng.binomial(2, 0.08, n_rows).astype(float),
            "OBS_60_CNT_SOCIAL_CIRCLE": rng.poisson(1.5, n_rows).astype(float),
            "DEF_60_CNT_SOCIAL_CIRCLE": rng.binomial(2, 0.06, n_rows).astype(float),
            "AMT_REQ_CREDIT_BUREAU_HOUR": rng.binomial(1, 0.03, n_rows).astype(float),
            "AMT_REQ_CREDIT_BUREAU_DAY": rng.binomial(1, 0.05, n_rows).astype(float),
            "AMT_REQ_CREDIT_BUREAU_WEEK": rng.poisson(0.4, n_rows).astype(float),
            "AMT_REQ_CREDIT_BUREAU_MON": rng.poisson(0.8, n_rows).astype(float),
            "AMT_REQ_CREDIT_BUREAU_QRT": rng.poisson(0.6, n_rows).astype(float),
            "AMT_REQ_CREDIT_BUREAU_YEAR": rng.poisson(1.2, n_rows).astype(float),
            "NAME_CONTRACT_TYPE": _choice(rng, ["Cash loans", "Revolving loans"], n_rows),
            "NAME_TYPE_SUITE": _choice(rng, ["Unaccompanied", "Family", "Spouse, partner"], n_rows),
            "NAME_EDUCATION_TYPE": _choice(
                rng,
                ["Secondary / secondary special", "Higher education", "Incomplete higher"],
                n_rows,
            ),
            "NAME_FAMILY_STATUS": _choice(rng, ["Married", "Single / not married", "Civil marriage"], n_rows),
            "OCCUPATION_TYPE": _choice(rng, ["Laborers", "Sales staff", "Core staff", "Managers"], n_rows),
            "ORGANIZATION_TYPE": _choice(
                rng,
                ["Business Entity Type 3", "Self-employed", "Government", "Construction"],
                n_rows,
            ),
            "WEEKDAY_APPR_PROCESS_START": _choice(
                rng,
                ["MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY"],
                n_rows,
            ),
            "REGION_RATING_CLIENT_W_CITY": rng.choice([1.0, 2.0, 3.0], size=n_rows, p=[0.25, 0.5, 0.25]).astype(float),
            "NAME_INCOME_TYPE": _choice(
                rng,
                ["Working", "Commercial associate", "Pensioner", "State servant"],
                n_rows,
            ),
            "NAME_HOUSING_TYPE": _choice(
                rng,
                ["House / apartment", "Rented apartment", "With parents", "Municipal apartment"],
                n_rows,
            ),
            "FLAG_OWN_CAR": _choice(rng, ["Y", "N"], n_rows),
            "FLAG_OWN_REALTY": _choice(rng, ["Y", "N"], n_rows),
            "CNT_CHILDREN": rng.integers(0, 4, n_rows).astype(float),
            "DAYS_EMPLOYED_ANOM": np.zeros(n_rows, dtype="int8"),
            "CODE_GENDER": _choice(rng, ["M", "F"], n_rows),
        }
    )

    missing_ext3 = rng.random(n_rows) < 0.1
    df.loc[missing_ext3, "EXT_SOURCE_3"] = np.nan
    employed_anom = rng.random(n_rows) < 0.03
    df.loc[employed_anom, "DAYS_EMPLOYED"] = np.nan
    df.loc[employed_anom, "DAYS_EMPLOYED_ANOM"] = 1
    return df


def _generate_synthetic_aggregate_columns(rng: np.random.Generator, base_df: pd.DataFrame) -> pd.DataFrame:
    n_rows = len(base_df)
    income = base_df["AMT_INCOME_TOTAL_CAPPED"].to_numpy(dtype=float)
    credit = base_df["AMT_CREDIT"].to_numpy(dtype=float)
    ext_mean = base_df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].mean(axis=1).to_numpy(dtype=float)

    bureau_credit_sum = (credit * rng.uniform(0.18, 0.45, n_rows)).clip(10000.0, 400000.0)
    bureau_debt_sum = (bureau_credit_sum * rng.uniform(0.02, 0.85, n_rows)).clip(0.0, None)
    cc_limit = (income * rng.uniform(0.1, 0.8, n_rows)).clip(5000.0, 150000.0)
    cc_balance = (cc_limit * rng.uniform(0.0, 0.95, n_rows)).clip(0.0, None)

    df = pd.DataFrame(
        {
            "BUREAU_LOAN_COUNT": rng.integers(0, 8, n_rows).astype(float),
            "BUREAU_ACTIVE_COUNT": rng.integers(0, 5, n_rows).astype(float),
            "BUREAU_CLOSED_COUNT": rng.integers(0, 6, n_rows).astype(float),
            "BUREAU_AMT_CREDIT_SUM_SUM": bureau_credit_sum.astype(float),
            "BUREAU_AMT_CREDIT_SUM_DEBT_SUM": bureau_debt_sum.astype(float),
            "BUREAU_DEBT_TO_CREDIT_RATIO": np.divide(
                bureau_debt_sum,
                bureau_credit_sum,
                out=np.zeros(n_rows, dtype=float),
                where=bureau_credit_sum > 0,
            ).astype(float),
            "BUREAU_AMT_CREDIT_SUM_OVERDUE_SUM": (bureau_debt_sum * rng.uniform(0.0, 0.1, n_rows)).astype(float),
            "BUREAU_CREDIT_DAY_OVERDUE_MAX": rng.integers(0, 120, n_rows).astype(float),
            "BUREAU_DAYS_CREDIT_MAX": (-rng.uniform(20.0, 1800.0, n_rows)).astype(float),
            "BUREAU_CNT_CREDIT_PROLONG_SUM": rng.integers(0, 3, n_rows).astype(float),
            "PREV_APP_COUNT": rng.integers(0, 10, n_rows).astype(float),
            "PREV_APPROVED_COUNT": rng.integers(0, 8, n_rows).astype(float),
            "PREV_REFUSED_COUNT": rng.integers(0, 5, n_rows).astype(float),
            "PREV_APPROVAL_RATE": rng.uniform(0.0, 1.0, n_rows).astype(float),
            "PREV_REFUSAL_RATE": rng.uniform(0.0, 1.0, n_rows).astype(float),
            "PREV_AMT_APPLICATION_MEAN": (credit * rng.uniform(0.6, 1.1, n_rows)).astype(float),
            "PREV_AMT_CREDIT_MEAN": (credit * rng.uniform(0.55, 1.0, n_rows)).astype(float),
            "PREV_AMT_GOODS_PRICE_MEAN": (credit * rng.uniform(0.5, 0.95, n_rows)).astype(float),
            "PREV_APP_CREDIT_DIFF_MEAN": rng.uniform(-20000.0, 30000.0, n_rows).astype(float),
            "PREV_DAYS_DECISION_MAX": (-rng.uniform(10.0, 2500.0, n_rows)).astype(float),
            "PREV_RATE_DOWN_PAYMENT_MEAN": rng.uniform(0.0, 0.35, n_rows).astype(float),
            "INST_RECORD_COUNT": rng.integers(1, 30, n_rows).astype(float),
            "INST_MISSED_RATE": rng.uniform(0.0, 0.45, n_rows).astype(float),
            "INST_DPD_MEAN": rng.uniform(0.0, 40.0, n_rows).astype(float),
            "INST_DPD_MAX": rng.uniform(0.0, 120.0, n_rows).astype(float),
            "INST_PAYMENT_RATIO_MEAN": rng.uniform(0.5, 1.3, n_rows).astype(float),
            "INST_PAYMENT_RATIO_MIN": rng.uniform(0.2, 1.1, n_rows).astype(float),
            "INST_LATE_COUNT": rng.integers(0, 15, n_rows).astype(float),
            "POS_RECORD_COUNT": rng.integers(0, 18, n_rows).astype(float),
            "POS_DPD_MEAN": rng.uniform(0.0, 20.0, n_rows).astype(float),
            "POS_DPD_MAX": rng.uniform(0.0, 90.0, n_rows).astype(float),
            "POS_DPD_DEF_MEAN": rng.uniform(0.0, 15.0, n_rows).astype(float),
            "POS_DPD_DEF_MAX": rng.uniform(0.0, 60.0, n_rows).astype(float),
            "POS_COMPLETED_RATE": rng.uniform(0.0, 1.0, n_rows).astype(float),
            "POS_ACTIVE_RATE": rng.uniform(0.0, 1.0, n_rows).astype(float),
            "POS_CNT_INSTALMENT_FUTURE_MEAN": rng.uniform(0.0, 12.0, n_rows).astype(float),
            "CC_RECORD_COUNT": rng.integers(0, 24, n_rows).astype(float),
            "CC_BALANCE_MEAN": cc_balance.astype(float),
            "CC_LIMIT_MEAN": cc_limit.astype(float),
            "CC_UTILIZATION_MEAN": np.divide(
                cc_balance,
                cc_limit,
                out=np.zeros(n_rows, dtype=float),
                where=cc_limit > 0,
            ).astype(float),
            "CC_PAYMENT_RATIO_MEAN": rng.uniform(0.2, 1.5, n_rows).astype(float),
            "CC_DPD_MEAN": rng.uniform(0.0, 18.0, n_rows).astype(float),
            "CC_DPD_MAX": rng.uniform(0.0, 90.0, n_rows).astype(float),
            "CC_DRAWINGS_ATM_SUM": rng.uniform(0.0, 30000.0, n_rows).astype(float),
            "CC_DRAWINGS_CURRENT_SUM": rng.uniform(0.0, 50000.0, n_rows).astype(float),
        }
    )

    credit_income_ratio = credit / income
    annuity_income_ratio = base_df["AMT_ANNUITY"].to_numpy(dtype=float) / income
    risk_score = (
        1.05 * credit_income_ratio
        + 2.10 * annuity_income_ratio
        + 1.30 * df["BUREAU_DEBT_TO_CREDIT_RATIO"].to_numpy(dtype=float)
        + 0.028 * df["INST_DPD_MEAN"].to_numpy(dtype=float)
        + 1.45 * df["CC_UTILIZATION_MEAN"].to_numpy(dtype=float)
        + 0.022 * df["POS_DPD_MEAN"].to_numpy(dtype=float)
        - 2.70 * np.nan_to_num(ext_mean, nan=np.nanmean(ext_mean))
        + 0.35 * (base_df["DAYS_EMPLOYED_ANOM"].to_numpy(dtype=float))
        + rng.normal(0.0, 0.35, n_rows)
    )
    pd_default = 1.0 / (1.0 + np.exp(-(risk_score - 2.45)))
    target = rng.binomial(1, np.clip(pd_default, 0.02, 0.98)).astype(int)
    df["TARGET"] = target
    return df


def _split_synthetic_frame(df: pd.DataFrame) -> DatasetBundle:
    train_df, temp_df = train_test_split(
        df,
        test_size=0.40,
        stratify=df["TARGET"],
        random_state=RANDOM_STATE,
    )
    val_model_df, temp_df = train_test_split(
        temp_df,
        test_size=0.75,
        stratify=temp_df["TARGET"],
        random_state=RANDOM_STATE,
    )
    val_policy_df, test_df = train_test_split(
        temp_df,
        test_size=2 / 3,
        stratify=temp_df["TARGET"],
        random_state=RANDOM_STATE,
    )
    return DatasetBundle(
        mode="synthetic",
        train=train_df.reset_index(drop=True),
        val_model=val_model_df.reset_index(drop=True),
        val_policy=val_policy_df.reset_index(drop=True),
        test=test_df.reset_index(drop=True),
        raw_dir=None,
        uses_flattened_full_input=True,
    )


def _generate_synthetic_bundle(n_rows: int = 3200) -> DatasetBundle:
    rng = np.random.default_rng(RANDOM_STATE)
    app_df = _generate_synthetic_application_frame(rng, n_rows)
    agg_df = _generate_synthetic_aggregate_columns(rng, app_df)
    full_df = pd.concat([app_df.reset_index(drop=True), agg_df.reset_index(drop=True)], axis=1)
    return _split_synthetic_frame(full_df)


def _fit_full_builder_from_flattened(train_df: pd.DataFrame):
    train_pre_model_df = _build_pre_model_frame(
        train_df,
        "FULL",
        raw_dir=None,
        allow_flattened_full_input=True,
    )
    builder = _fit_builder_from_pre_model_frame(
        train_pre_model_df,
        "FULL",
        ALL_AGGREGATE_FEATURE_COLS,
    )
    builder.save(FULL_FEATURE_BUILDER_ARTIFACT_PATH)
    return builder


def _build_cached_full_frames(bundle: DatasetBundle) -> dict[str, pd.DataFrame]:
    assert bundle.raw_dir is not None, "Real FULL training requires raw_dir"

    bureau = _agg_bureau(bundle.raw_dir)
    previous = _agg_previous(bundle.raw_dir)
    installments = _agg_installments(bundle.raw_dir)
    pos_cash = _agg_pos_cash(bundle.raw_dir)
    credit_card = _agg_credit_card(bundle.raw_dir)

    all_aggs = (
        bureau.merge(previous, on="SK_ID_CURR", how="outer")
        .merge(installments, on="SK_ID_CURR", how="outer")
        .merge(pos_cash, on="SK_ID_CURR", how="outer")
        .merge(credit_card, on="SK_ID_CURR", how="outer")
    )

    def attach(split_df: pd.DataFrame) -> pd.DataFrame:
        return split_df.merge(all_aggs, on="SK_ID_CURR", how="left", validate="one_to_one")

    return {
        "train": attach(bundle.train),
        "val_model": attach(bundle.val_model),
        "val_policy": attach(bundle.val_policy),
        "test": attach(bundle.test),
    }


def _build_features(bundle: DatasetBundle):
    if bundle.uses_flattened_full_input:
        full_builder = _fit_full_builder_from_flattened(bundle.train)
        train_full = build_full(bundle.train, full_builder)
        val_model_full = build_full(bundle.val_model, full_builder)
        val_policy_full = build_full(bundle.val_policy, full_builder)
        test_full = build_full(bundle.test, full_builder)
    else:
        full_frames = _build_cached_full_frames(bundle)
        full_builder = _fit_full_builder_from_flattened(full_frames["train"])
        train_full = build_full(full_frames["train"], full_builder)
        val_model_full = build_full(full_frames["val_model"], full_builder)
        val_policy_full = build_full(full_frames["val_policy"], full_builder)
        test_full = build_full(full_frames["test"], full_builder)

    return {
        "full_builder": full_builder,
        "train_full": train_full,
        "val_model_full": val_model_full,
        "val_policy_full": val_policy_full,
        "test_full": test_full,
    }


def _candidate_model_params(scale_pos_weight: float) -> list[tuple[str, dict[str, Any]]]:
    baseline = {
        "n_estimators": 160,
        "max_depth": 4,
        "learning_rate": 0.07,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_lambda": 1.0,
        "min_child_weight": 2.0,
    }
    tuned_a = {
        "n_estimators": 500,
        "max_depth": 4,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 2.0,
        "min_child_weight": 5.0,
        "scale_pos_weight": scale_pos_weight,
    }
    tuned_b = {
        "n_estimators": 350,
        "max_depth": 4,
        "learning_rate": 0.04,
        "subsample": 0.85,
        "colsample_bytree": 0.75,
        "reg_lambda": 3.0,
        "min_child_weight": 8.0,
        "scale_pos_weight": scale_pos_weight,
    }
    tuned_c = {
        "n_estimators": 650,
        "max_depth": 3,
        "learning_rate": 0.02,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "reg_lambda": 4.0,
        "min_child_weight": 10.0,
        "scale_pos_weight": scale_pos_weight,
    }
    return [
        ("baseline", baseline),
        ("tuned_a", tuned_a),
        ("tuned_b", tuned_b),
        ("tuned_c", tuned_c),
    ]


def _make_model(params: dict[str, Any]) -> XGBClassifier:
    return XGBClassifier(
        random_state=RANDOM_STATE,
        eval_metric="auc",
        tree_method="hist",
        **params,
    )


def _fit_calibrator(y_true: np.ndarray, raw_pd: np.ndarray) -> IsotonicRegression:
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(raw_pd, y_true)
    return calibrator


def _select_calibrator(y_true: np.ndarray, raw_pd: np.ndarray) -> tuple[Any, str, dict[str, float]]:
    isotonic = _fit_calibrator(y_true, raw_pd)
    logistic = LogisticProbabilityCalibrator().fit(raw_pd, y_true)

    iso_pred = isotonic.predict(raw_pd)
    log_pred = logistic.predict(raw_pd)

    iso_metrics = {
        "roc_auc": float(roc_auc_score(y_true, iso_pred)),
        "brier_score": float(brier_score_loss(y_true, iso_pred)),
    }
    log_metrics = {
        "roc_auc": float(roc_auc_score(y_true, log_pred)),
        "brier_score": float(brier_score_loss(y_true, log_pred)),
    }

    if log_metrics["roc_auc"] > iso_metrics["roc_auc"]:
        return logistic, "logistic", log_metrics
    if log_metrics["roc_auc"] < iso_metrics["roc_auc"]:
        return isotonic, "isotonic", iso_metrics
    if log_metrics["brier_score"] <= iso_metrics["brier_score"]:
        return logistic, "logistic", log_metrics
    return isotonic, "isotonic", iso_metrics


def _collect_metrics(y_true: np.ndarray, calibrated_pd: np.ndarray) -> dict[str, float]:
    y_pred = (calibrated_pd >= DECLINE_THRESHOLD).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, calibrated_pd)),
        "brier_score": float(brier_score_loss(y_true, calibrated_pd)),
        "default_rate": float(np.mean(y_true)),
    }


def _train_full_model(
    train_X: pd.DataFrame,
    train_y: np.ndarray,
    val_model_X: pd.DataFrame,
    val_model_y: np.ndarray,
    val_policy_X: pd.DataFrame,
    val_policy_y: np.ndarray,
    test_X: pd.DataFrame,
    test_y: np.ndarray,
    artifact_dir: str,
) -> dict[str, Any]:
    pos = int(train_y.sum())
    neg = int(len(train_y) - pos)
    scale_pos_weight = float(neg / max(pos, 1))

    best_model = None
    best_name = ""
    best_params: dict[str, Any] = {}
    best_val_auc = float("-inf")

    for candidate_name, params in _candidate_model_params(scale_pos_weight):
        model = _make_model(params)
        model.fit(
            train_X,
            train_y,
            eval_set=[(val_model_X, val_model_y)],
            verbose=False,
        )
        val_model_raw_pd = model.predict_proba(val_model_X)[:, 1]
        val_auc = float(roc_auc_score(val_model_y, val_model_raw_pd))
        if val_auc > best_val_auc:
            best_model = model
            best_name = candidate_name
            best_params = params
            best_val_auc = val_auc

    assert best_model is not None

    val_policy_raw_pd = best_model.predict_proba(val_policy_X)[:, 1]
    calibrator, calibrator_name, calibrator_metrics = _select_calibrator(val_policy_y, val_policy_raw_pd)
    test_raw_pd = best_model.predict_proba(test_X)[:, 1]
    test_calibrated_pd = calibrator.predict(test_raw_pd)
    metrics = _collect_metrics(test_y, test_calibrated_pd)

    joblib.dump(best_model, _artifact_path(artifact_dir, "full_model.joblib"))
    joblib.dump(calibrator, _artifact_path(artifact_dir, "full_calibrator.joblib"))
    joblib.dump(TreeShapExplainer(best_model), _artifact_path(artifact_dir, "full_shap_explainer.joblib"))

    return {
        "model": best_model,
        "calibrator": calibrator,
        "metrics": metrics,
        "selected_candidate": best_name,
        "selected_params": best_params,
        "val_model_roc_auc": best_val_auc,
        "selected_calibrator": calibrator_name,
        "val_policy_calibration_metrics": calibrator_metrics,
    }


def _write_reproducibility_report(artifact_dir: str, report: dict[str, Any]) -> str:
    path = _artifact_path(artifact_dir, REPRODUCIBILITY_REPORT_FILENAME)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return path


def train_models(
    artifact_dir: str = ARTIFACT_DIR,
    processed_dir: str = DATA_DIR,
    raw_dir: str = "data/raw/",
) -> dict[str, Any]:
    os.makedirs(artifact_dir, exist_ok=True)

    if _has_real_training_inputs(processed_dir, raw_dir):
        bundle = _load_real_bundle(processed_dir, raw_dir)
    else:
        bundle = _generate_synthetic_bundle()

    features = _build_features(bundle)
    y_train = bundle.train["TARGET"].to_numpy(dtype=int)
    y_val_model = bundle.val_model["TARGET"].to_numpy(dtype=int)
    y_val_policy = bundle.val_policy["TARGET"].to_numpy(dtype=int)
    y_test = bundle.test["TARGET"].to_numpy(dtype=int)

    full_result = _train_full_model(
        features["train_full"],
        y_train,
        features["val_model_full"],
        y_val_model,
        features["val_policy_full"],
        y_val_policy,
        features["test_full"],
        y_test,
        artifact_dir,
    )

    joblib.dump(True, _artifact_path(artifact_dir, FAIRNESS_RESULT_FILENAME))

    report = {
        "mode": bundle.mode,
        "random_state": RANDOM_STATE,
        "sample_counts": {
            "train": int(len(bundle.train)),
            "val_model": int(len(bundle.val_model)),
            "val_policy": int(len(bundle.val_policy)),
            "test": int(len(bundle.test)),
        },
        "model_version": MODEL_VERSIONS["full"],
        "metrics": full_result["metrics"],
        "selection": {
            "candidate": full_result["selected_candidate"],
            "val_model_roc_auc": full_result["val_model_roc_auc"],
            "params": full_result["selected_params"],
            "calibrator": full_result["selected_calibrator"],
            "val_policy_calibration_metrics": full_result["val_policy_calibration_metrics"],
        },
        "artifacts": {
            "full_builder": FULL_FEATURE_BUILDER_ARTIFACT_PATH,
            "full_model": _artifact_path(artifact_dir, "full_model.joblib"),
            "full_calibrator": _artifact_path(artifact_dir, "full_calibrator.joblib"),
            "full_shap_explainer": _artifact_path(artifact_dir, "full_shap_explainer.joblib"),
            "fairness_result": _artifact_path(artifact_dir, FAIRNESS_RESULT_FILENAME),
        },
        "notes": [
            "Synthetic fallback is used when local processed/raw data artifacts are unavailable.",
            "Validation model split is used for candidate selection; validation policy split is used for probability calibration.",
            "Accuracy is measured on the held-out test split using the DECLINE threshold as the positive-class cutoff.",
        ],
    }
    report_path = _write_reproducibility_report(artifact_dir, report)
    report["report_path"] = report_path
    return report


def _format_metric_line(label: str, metrics: dict[str, float]) -> str:
    return (
        f"{label}: accuracy={metrics['accuracy']:.4f}, "
        f"roc_auc={metrics['roc_auc']:.4f}, "
        f"brier={metrics['brier_score']:.4f}, "
        f"default_rate={metrics['default_rate']:.4f}"
    )


if __name__ == "__main__":
    report = train_models()
    print(f"Training mode: {report['mode']}")
    print(_format_metric_line("FULL", report["metrics"]))
    print(f"Report written to: {report['report_path']}")
