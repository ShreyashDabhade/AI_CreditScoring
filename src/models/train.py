"""Module 3 â€” Model Training.

Trains a FULL scoring model, calibrates probabilities, persists artifacts
for Module 5, and writes a reproducibility report.

If real processed/raw data is unavailable in the local clone, this module falls
back to a deterministic synthetic dataset so the branch remains runnable.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from dataclasses import dataclass
from typing import Any

if __package__ is None or __package__ == "":
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.builder_artifacts import load_validated_builders

from configs.config import (
    APPROVE_THRESHOLD,
    ARTIFACT_DIR,
    DATA_DIR,
    DECLINE_THRESHOLD,
    MODEL_VERSIONS,
    RANDOM_STATE,
)
from src.feature_engineering import (
    DEFAULT_FULL_FEATURE_VIEW,
    FORBIDDEN_COLS,
    FULL_FEATURE_BUILDER_ARTIFACT_PATH,
    REDUCED_FEATURE_BUILDER_ARTIFACT_PATH,
    _aggregate_contract_version_for_view,
    _agg_bureau,
    _agg_credit_card,
    _agg_installments,
    _agg_pos_cash,
    _agg_previous,
    _build_pre_model_frame,
    _fit_builder_from_pre_model_frame,
    build_full,
    build_reduced,
    fit_reduced_builder,
    resolve_full_feature_view,
)
from src.models.runtime_support import (
    LogisticProbabilityCalibrator,
    TreeShapExplainer,
    WeightedBlendModel,
    WeightedBlendShapExplainer,
)
from src.runtime_verification import (
    dataframe_fingerprint,
    load_json_object,
    validate_transformed_frame,
)

REPRODUCIBILITY_REPORT_FILENAME = "reproducibility_report.json"
META_BLEND_EXPERIMENT_REPORT_FILENAME = "meta_blend_experiment_report.json"
FAIRNESS_RESULT_FILENAME = "model_fairness_audit_passed.joblib"
PROCESSED_MANIFEST_FILENAME = "processed_artifact_manifest.json"
RAW_TABLE_NAMES = [
    "bureau.csv",
    "previous_application.csv",
    "installments_payments.csv",
    "POS_CASH_balance.csv",
    "credit_card_balance.csv",
]
FULL_WEIGHTED_BLEND_MODEL_VERSION = "full_weighted_blend_v2.2.0"
FULL_XGBOOST_CANDIDATE_PREFIX = "full_xgboost"
FULL_WEIGHTED_BLEND_CANDIDATE_PREFIX = "full_weighted_blend"
FULL_WEIGHTED_BLEND_METADATA_FILENAME = "full_weighted_blend_metadata.json"


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


def _builder_artifact_path(artifact_dir: str, tier: str) -> str:
    filename = os.path.basename(
        FULL_FEATURE_BUILDER_ARTIFACT_PATH if tier.upper() == "FULL" else REDUCED_FEATURE_BUILDER_ARTIFACT_PATH
    )
    return _artifact_path(artifact_dir, filename)


def _processed_manifest_path(processed_dir: str) -> str:
    return os.path.join(processed_dir, PROCESSED_MANIFEST_FILENAME)


def _processed_manifest_fingerprint(processed_dir: str) -> str | None:
    manifest_path = _processed_manifest_path(processed_dir)
    if not os.path.exists(manifest_path):
        return None
    manifest = load_json_object(manifest_path, "processed manifest")
    fingerprint = manifest.get("processed_manifest_fingerprint")
    return str(fingerprint) if fingerprint else None


def load_artifacts(
    artifact_dir: str = ARTIFACT_DIR,
    processed_dir: str = DATA_DIR,
    strict_artifacts: bool = False,
) -> dict[str, Any]:
    builders = load_validated_builders(
        artifact_dir=artifact_dir,
        processed_dir=processed_dir,
        strict_artifacts=strict_artifacts,
    )
    report_path = _artifact_path(artifact_dir, REPRODUCIBILITY_REPORT_FILENAME)
    reproducibility_report: dict[str, Any] = {}
    if os.path.exists(report_path):
        reproducibility_report = load_json_object(report_path, "reproducibility report")
    fairness_value = joblib.load(_artifact_path(artifact_dir, FAIRNESS_RESULT_FILENAME))
    if isinstance(fairness_value, np.bool_):
        fairness_value = bool(fairness_value)
    if not isinstance(fairness_value, bool):
        raise RuntimeError("model_fairness_audit_passed.joblib must contain a boolean placeholder/result")
    return {
        "full_model": joblib.load(_artifact_path(artifact_dir, "full_model.joblib")),
        "full_calibrator": joblib.load(_artifact_path(artifact_dir, "full_calibrator.joblib")),
        "full_shap_explainer": joblib.load(_artifact_path(artifact_dir, "full_shap_explainer.joblib")),
        "full_builder": builders["FULL"],
        "reduced_model": joblib.load(_artifact_path(artifact_dir, "reduced_model.joblib")),
        "reduced_calibrator": joblib.load(_artifact_path(artifact_dir, "reduced_calibrator.joblib")),
        "reduced_shap_explainer": joblib.load(_artifact_path(artifact_dir, "reduced_shap_explainer.joblib")),
        "reduced_builder": builders["REDUCED"],
        "model_fairness_audit_passed": fairness_value,
        "reproducibility_report": reproducibility_report,
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


def _fit_full_builder_from_flattened(
    train_df: pd.DataFrame,
    *,
    source_df: pd.DataFrame | None = None,
    save_path: str | None = None,
    processed_manifest_fingerprint: str | None = None,
    feature_view: str = DEFAULT_FULL_FEATURE_VIEW,
):
    source_frame = source_df if source_df is not None else train_df
    view_name, _, aggregate_feature_cols = resolve_full_feature_view(feature_view)
    train_pre_model_df = _build_pre_model_frame(
        train_df,
        "FULL",
        raw_dir=None,
        allow_flattened_full_input=True,
        feature_view=view_name,
    )
    builder = _fit_builder_from_pre_model_frame(
        train_pre_model_df,
        "FULL",
        aggregate_feature_cols,
        dataset_fingerprint=dataframe_fingerprint(source_frame),
        fit_split_name="train",
        processed_manifest_fingerprint=processed_manifest_fingerprint,
        aggregate_contract_version=_aggregate_contract_version_for_view(view_name, aggregate_feature_cols),
    )
    if save_path is not None:
        builder.save(save_path)
    return builder


def _fit_reduced_builder_from_application(
    train_df: pd.DataFrame,
    *,
    save_path: str | None = None,
    processed_manifest_path: str | None = None,
):
    return fit_reduced_builder(
        train_df,
        save_path=save_path,
        processed_manifest_path=processed_manifest_path,
    )


def _build_cached_full_frames(
    bundle: DatasetBundle,
    feature_view: str = DEFAULT_FULL_FEATURE_VIEW,
) -> dict[str, pd.DataFrame]:
    assert bundle.raw_dir is not None, "Real FULL training requires raw_dir"

    _, families, _ = resolve_full_feature_view(feature_view)
    family_frames: list[pd.DataFrame] = []
    if "BUREAU" in families:
        family_frames.append(_agg_bureau(bundle.raw_dir))
    if "PREVIOUS_APPLICATION" in families:
        family_frames.append(_agg_previous(bundle.raw_dir))
    if "INSTALLMENTS" in families:
        family_frames.append(_agg_installments(bundle.raw_dir))
    if "POS_CASH" in families:
        family_frames.append(_agg_pos_cash(bundle.raw_dir))
    if "CREDIT_CARD" in families:
        family_frames.append(_agg_credit_card(bundle.raw_dir))

    all_aggs = family_frames[0]
    for feat_df in family_frames[1:]:
        all_aggs = all_aggs.merge(feat_df, on="SK_ID_CURR", how="outer")

    def attach(split_df: pd.DataFrame) -> pd.DataFrame:
        return split_df.merge(all_aggs, on="SK_ID_CURR", how="left", validate="one_to_one")

    return {
        "train": attach(bundle.train),
        "val_model": attach(bundle.val_model),
        "val_policy": attach(bundle.val_policy),
        "test": attach(bundle.test),
    }


def _build_features(
    bundle: DatasetBundle,
    artifact_dir: str,
    processed_dir: str,
    full_feature_view: str = DEFAULT_FULL_FEATURE_VIEW,
):
    processed_manifest_path = _processed_manifest_path(processed_dir)
    reduced_builder = _fit_reduced_builder_from_application(
        bundle.train,
        save_path=_builder_artifact_path(artifact_dir, "REDUCED"),
        processed_manifest_path=processed_manifest_path if os.path.exists(processed_manifest_path) else None,
    )
    train_reduced = build_reduced(bundle.train, reduced_builder)
    val_model_reduced = build_reduced(bundle.val_model, reduced_builder)
    val_policy_reduced = build_reduced(bundle.val_policy, reduced_builder)
    test_reduced = build_reduced(bundle.test, reduced_builder)

    processed_manifest_fingerprint = _processed_manifest_fingerprint(processed_dir)
    if bundle.uses_flattened_full_input:
        full_builder = _fit_full_builder_from_flattened(
            bundle.train,
            source_df=bundle.train,
            save_path=_builder_artifact_path(artifact_dir, "FULL"),
            processed_manifest_fingerprint=processed_manifest_fingerprint,
            feature_view=full_feature_view,
        )
        train_full = build_full(bundle.train, full_builder, feature_view=full_feature_view)
        val_model_full = build_full(bundle.val_model, full_builder, feature_view=full_feature_view)
        val_policy_full = build_full(bundle.val_policy, full_builder, feature_view=full_feature_view)
        test_full = build_full(bundle.test, full_builder, feature_view=full_feature_view)
    else:
        full_frames = _build_cached_full_frames(bundle, full_feature_view)
        full_builder = _fit_full_builder_from_flattened(
            full_frames["train"],
            source_df=bundle.train,
            save_path=_builder_artifact_path(artifact_dir, "FULL"),
            processed_manifest_fingerprint=processed_manifest_fingerprint,
            feature_view=full_feature_view,
        )
        train_full = build_full(full_frames["train"], full_builder, feature_view=full_feature_view)
        val_model_full = build_full(full_frames["val_model"], full_builder, feature_view=full_feature_view)
        val_policy_full = build_full(full_frames["val_policy"], full_builder, feature_view=full_feature_view)
        test_full = build_full(full_frames["test"], full_builder, feature_view=full_feature_view)

    for frame_name, frame, builder in [
        ("train_full", train_full, full_builder),
        ("val_model_full", val_model_full, full_builder),
        ("val_policy_full", val_policy_full, full_builder),
        ("test_full", test_full, full_builder),
        ("train_reduced", train_reduced, reduced_builder),
        ("val_model_reduced", val_model_reduced, reduced_builder),
        ("val_policy_reduced", val_policy_reduced, reduced_builder),
        ("test_reduced", test_reduced, reduced_builder),
    ]:
        validate_transformed_frame(
            frame,
            expected_columns=builder.encoded_columns_,
            forbidden_columns=FORBIDDEN_COLS,
        )

    return {
        "full_builder": full_builder,
        "reduced_builder": reduced_builder,
        "train_full": train_full,
        "val_model_full": val_model_full,
        "val_policy_full": val_policy_full,
        "test_full": test_full,
        "train_reduced": train_reduced,
        "val_model_reduced": val_model_reduced,
        "val_policy_reduced": val_policy_reduced,
        "test_reduced": test_reduced,
    }


def _candidate_lgbm_params(scale_pos_weight: float) -> list[tuple[str, dict[str, Any]]]:
    baseline = {
        "n_estimators": 220,
        "num_leaves": 31,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_lambda": 1.0,
        "min_child_samples": 60,
        "scale_pos_weight": scale_pos_weight,
    }
    tuned_a = {
        "n_estimators": 420,
        "num_leaves": 31,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 2.0,
        "min_child_samples": 90,
        "scale_pos_weight": scale_pos_weight,
    }
    tuned_b = {
        "n_estimators": 360,
        "num_leaves": 63,
        "learning_rate": 0.04,
        "subsample": 0.85,
        "colsample_bytree": 0.75,
        "reg_lambda": 3.0,
        "min_child_samples": 120,
        "scale_pos_weight": scale_pos_weight,
    }
    tuned_c = {
        "n_estimators": 520,
        "num_leaves": 63,
        "learning_rate": 0.025,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "reg_lambda": 4.0,
        "min_child_samples": 160,
        "scale_pos_weight": scale_pos_weight,
    }
    return [
        ("baseline", baseline),
        ("tuned_a", tuned_a),
        ("tuned_b", tuned_b),
        ("tuned_c", tuned_c),
    ]


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


def _make_lgbm_model(params: dict[str, Any]) -> LGBMClassifier:
    return LGBMClassifier(
        objective="binary",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
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


def _train_tier_model(
    tier: str,
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
    tier_prefix = tier.lower()
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

    joblib.dump(best_model, _artifact_path(artifact_dir, f"{tier_prefix}_model.joblib"))
    joblib.dump(calibrator, _artifact_path(artifact_dir, f"{tier_prefix}_calibrator.joblib"))
    joblib.dump(
        TreeShapExplainer(best_model),
        _artifact_path(artifact_dir, f"{tier_prefix}_shap_explainer.joblib"),
    )

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


def _train_best_model_family(
    model_family: str,
    candidate_params: list[tuple[str, dict[str, Any]]],
    train_X: pd.DataFrame,
    train_y: np.ndarray,
    val_model_X: pd.DataFrame,
    val_model_y: np.ndarray,
) -> dict[str, Any]:
    best_model = None
    best_name = ""
    best_params: dict[str, Any] = {}
    best_val_auc = float("-inf")
    best_val_model_raw_pd: np.ndarray | None = None

    for candidate_name, params in candidate_params:
        if model_family == "xgboost":
            model = _make_model(params)
            model.fit(train_X, train_y, eval_set=[(val_model_X, val_model_y)], verbose=False)
            val_model_raw_pd = model.predict_proba(val_model_X)[:, 1]
        elif model_family == "lightgbm":
            model = _make_lgbm_model(params)
            train_matrix = train_X.to_numpy(dtype=float)
            val_model_matrix = val_model_X.to_numpy(dtype=float)
            model.fit(train_matrix, train_y)
            val_model_raw_pd = model.predict_proba(val_model_matrix)[:, 1]
        else:
            raise ValueError(f"Unsupported model family: {model_family}")
        val_auc = float(roc_auc_score(val_model_y, val_model_raw_pd))
        if val_auc > best_val_auc:
            best_model = model
            best_name = candidate_name
            best_params = params
            best_val_auc = val_auc
            best_val_model_raw_pd = val_model_raw_pd

    assert best_model is not None and best_val_model_raw_pd is not None
    return {
        "model": best_model,
        "selected_candidate": best_name,
        "selected_params": best_params,
        "val_model_roc_auc": best_val_auc,
        "val_model_raw_pd": best_val_model_raw_pd,
    }


def _evaluate_calibrated_scores(
    val_policy_y: np.ndarray,
    val_policy_raw_pd: np.ndarray,
    test_y: np.ndarray,
    test_raw_pd: np.ndarray,
) -> dict[str, Any]:
    calibrator, calibrator_name, calibrator_metrics = _select_calibrator(val_policy_y, val_policy_raw_pd)
    test_calibrated_pd = calibrator.predict(test_raw_pd)
    return {
        "calibrator": calibrator,
        "selected_calibrator": calibrator_name,
        "val_policy_calibration_metrics": calibrator_metrics,
        "metrics": _collect_metrics(test_y, test_calibrated_pd),
    }


def _select_weighted_average_blend(
    y_true: np.ndarray,
    pred_xgb: np.ndarray,
    pred_lgbm: np.ndarray,
    weights: np.ndarray | None = None,
) -> dict[str, Any]:
    weight_grid = np.linspace(0.0, 1.0, 21) if weights is None else np.asarray(weights, dtype=float)
    best_weight = 0.0
    best_auc = float("-inf")
    best_pred = pred_lgbm
    for weight in weight_grid:
        blended = weight * pred_xgb + (1.0 - weight) * pred_lgbm
        auc = float(roc_auc_score(y_true, blended))
        if auc > best_auc:
            best_auc = auc
            best_weight = float(weight)
            best_pred = blended
    return {
        "weight_xgb": best_weight,
        "weight_lgbm": float(1.0 - best_weight),
        "val_model_roc_auc": best_auc,
        "val_model_raw_pd": best_pred,
    }


def _fit_logistic_meta_blend(
    y_true: np.ndarray,
    pred_xgb: np.ndarray,
    pred_lgbm: np.ndarray,
    *,
    regularization_c: float = 1.0,
) -> dict[str, Any]:
    meta_X = np.column_stack([pred_xgb, pred_lgbm])
    meta = LogisticRegression(
        solver="liblinear",
        random_state=RANDOM_STATE,
        C=float(regularization_c),
    )
    meta.fit(meta_X, y_true)
    raw_pd = meta.predict_proba(meta_X)[:, 1]
    return {
        "meta_model": meta,
        "regularization_c": float(regularization_c),
        "val_model_roc_auc": float(roc_auc_score(y_true, raw_pd)),
        "val_model_raw_pd": raw_pd,
    }


def _predict_lightgbm_raw_pd(model: Any, X: pd.DataFrame | np.ndarray) -> np.ndarray:
    matrix = X.to_numpy(dtype=float, copy=False) if hasattr(X, "to_numpy") else np.asarray(X, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    return np.asarray(model.predict_proba(matrix), dtype=float)[:, 1]


def _persist_runtime_bundle(
    artifact_dir: str,
    prefix: str,
    *,
    model: Any,
    calibrator: Any,
    explainer: Any,
) -> dict[str, str]:
    paths = {
        "model": _artifact_path(artifact_dir, f"{prefix}_model.joblib"),
        "calibrator": _artifact_path(artifact_dir, f"{prefix}_calibrator.joblib"),
        "explainer": _artifact_path(artifact_dir, f"{prefix}_shap_explainer.joblib"),
    }
    joblib.dump(model, paths["model"])
    joblib.dump(calibrator, paths["calibrator"])
    joblib.dump(explainer, paths["explainer"])
    return paths


def _write_json_artifact(artifact_dir: str, filename: str, payload: dict[str, Any]) -> str:
    path = _artifact_path(artifact_dir, filename)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return path


def _train_full_runtime_candidate(
    train_X: pd.DataFrame,
    train_y: np.ndarray,
    val_model_X: pd.DataFrame,
    val_model_y: np.ndarray,
    val_policy_X: pd.DataFrame,
    val_policy_y: np.ndarray,
    test_X: pd.DataFrame,
    test_y: np.ndarray,
    artifact_dir: str,
    *,
    full_feature_view: str,
    processed_manifest_fingerprint: str | None,
) -> dict[str, Any]:
    pos = int(train_y.sum())
    neg = int(len(train_y) - pos)
    scale_pos_weight = float(neg / max(pos, 1))

    xgb_result = _train_best_model_family(
        "xgboost",
        _candidate_model_params(scale_pos_weight),
        train_X,
        train_y,
        val_model_X,
        val_model_y,
    )
    lgbm_result = _train_best_model_family(
        "lightgbm",
        _candidate_lgbm_params(scale_pos_weight),
        train_X,
        train_y,
        val_model_X,
        val_model_y,
    )

    xgb_val_policy_raw = xgb_result["model"].predict_proba(val_policy_X)[:, 1]
    xgb_test_raw = xgb_result["model"].predict_proba(test_X)[:, 1]
    lgbm_val_policy_raw = _predict_lightgbm_raw_pd(lgbm_result["model"], val_policy_X)
    lgbm_test_raw = _predict_lightgbm_raw_pd(lgbm_result["model"], test_X)

    xgb_eval = _evaluate_calibrated_scores(val_policy_y, xgb_val_policy_raw, test_y, xgb_test_raw)
    weighted = _select_weighted_average_blend(
        val_model_y,
        xgb_result["val_model_raw_pd"],
        lgbm_result["val_model_raw_pd"],
    )
    weighted_model = WeightedBlendModel(
        xgb_result["model"],
        lgbm_result["model"],
        weight_xgboost=float(weighted["weight_xgb"]),
        weight_lightgbm=float(weighted["weight_lgbm"]),
        feature_count=int(train_X.shape[1]),
    )
    weighted_val_policy_raw = weighted_model.predict_raw_pd(val_policy_X)
    weighted_test_raw = weighted_model.predict_raw_pd(test_X)
    weighted_eval = _evaluate_calibrated_scores(
        val_policy_y,
        weighted_val_policy_raw,
        test_y,
        weighted_test_raw,
    )

    xgb_explainer = TreeShapExplainer(xgb_result["model"])
    lgbm_explainer = TreeShapExplainer(lgbm_result["model"])
    weighted_explainer = WeightedBlendShapExplainer(
        xgb_explainer,
        lgbm_explainer,
        weight_xgboost=float(weighted["weight_xgb"]),
        weight_lightgbm=float(weighted["weight_lgbm"]),
    )

    xgb_candidate_artifacts = _persist_runtime_bundle(
        artifact_dir,
        FULL_XGBOOST_CANDIDATE_PREFIX,
        model=xgb_result["model"],
        calibrator=xgb_eval["calibrator"],
        explainer=xgb_explainer,
    )
    weighted_candidate_artifacts = _persist_runtime_bundle(
        artifact_dir,
        FULL_WEIGHTED_BLEND_CANDIDATE_PREFIX,
        model=weighted_model,
        calibrator=weighted_eval["calibrator"],
        explainer=weighted_explainer,
    )

    xgb_candidate = {
        "model_family": "xgboost",
        "model_version": MODEL_VERSIONS["full"],
        "selected_candidate": xgb_result["selected_candidate"],
        "selected_params": xgb_result["selected_params"],
        "val_model_roc_auc": float(xgb_result["val_model_roc_auc"]),
        "test_metrics": xgb_eval["metrics"],
        "selected_calibrator": xgb_eval["selected_calibrator"],
        "val_policy_calibration_metrics": xgb_eval["val_policy_calibration_metrics"],
        "artifacts": xgb_candidate_artifacts,
        "explanation_policy": "tree_shap",
    }
    lightgbm_component = {
        "model_family": "lightgbm",
        "selected_candidate": lgbm_result["selected_candidate"],
        "selected_params": lgbm_result["selected_params"],
        "val_model_roc_auc": float(lgbm_result["val_model_roc_auc"]),
        "test_metrics": _evaluate_calibrated_scores(
            val_policy_y,
            lgbm_val_policy_raw,
            test_y,
            lgbm_test_raw,
        )["metrics"],
    }
    weighted_candidate = {
        "model_family": "weighted_blend",
        "model_version": FULL_WEIGHTED_BLEND_MODEL_VERSION,
        "blend_method": "weighted_average",
        "weights": {
            "xgboost": float(weighted["weight_xgb"]),
            "lightgbm": float(weighted["weight_lgbm"]),
        },
        "val_model_roc_auc": float(weighted["val_model_roc_auc"]),
        "test_metrics": weighted_eval["metrics"],
        "selected_calibrator": weighted_eval["selected_calibrator"],
        "val_policy_calibration_metrics": weighted_eval["val_policy_calibration_metrics"],
        "component_selection": {
            "xgboost": {
                "candidate": xgb_result["selected_candidate"],
                "params": xgb_result["selected_params"],
            },
            "lightgbm": {
                "candidate": lgbm_result["selected_candidate"],
                "params": lgbm_result["selected_params"],
            },
        },
        "artifacts": weighted_candidate_artifacts,
        "explanation_policy": weighted_explainer.explanation_policy,
    }

    selected_candidate = "weighted_blend_full" if weighted_candidate["val_model_roc_auc"] > xgb_candidate["val_model_roc_auc"] else "xgboost_full"
    if selected_candidate == "weighted_blend_full":
        selected_model = weighted_model
        selected_calibrator = weighted_eval["calibrator"]
        selected_explainer = weighted_explainer
        selected_metrics = weighted_eval["metrics"]
        selected_model_family = "weighted_blend"
        selected_model_version = FULL_WEIGHTED_BLEND_MODEL_VERSION
        selected_val_model_roc_auc = float(weighted_candidate["val_model_roc_auc"])
        selected_calibrator_name = weighted_eval["selected_calibrator"]
        selected_calibration_metrics = weighted_eval["val_policy_calibration_metrics"]
    else:
        selected_model = xgb_result["model"]
        selected_calibrator = xgb_eval["calibrator"]
        selected_explainer = xgb_explainer
        selected_metrics = xgb_eval["metrics"]
        selected_model_family = "xgboost"
        selected_model_version = MODEL_VERSIONS["full"]
        selected_val_model_roc_auc = float(xgb_candidate["val_model_roc_auc"])
        selected_calibrator_name = xgb_eval["selected_calibrator"]
        selected_calibration_metrics = xgb_eval["val_policy_calibration_metrics"]

    selected_runtime_artifacts = _persist_runtime_bundle(
        artifact_dir,
        "full",
        model=selected_model,
        calibrator=selected_calibrator,
        explainer=selected_explainer,
    )

    blend_metadata = {
        "processed_manifest_fingerprint": processed_manifest_fingerprint,
        "feature_view": full_feature_view,
        "blend_method": "weighted_average",
        "model_version": FULL_WEIGHTED_BLEND_MODEL_VERSION,
        "weights": weighted_candidate["weights"],
        "component_candidates": weighted_candidate["component_selection"],
        "selected_runtime_candidate": selected_candidate,
        "selected_runtime_model_version": selected_model_version,
        "selected_runtime_artifacts": selected_runtime_artifacts,
        "fallback_xgboost_artifacts": xgb_candidate_artifacts,
        "weighted_blend_artifacts": weighted_candidate_artifacts,
        "xgboost_candidate": xgb_candidate,
        "lightgbm_component": lightgbm_component,
        "weighted_blend_candidate": weighted_candidate,
        "explanation_policy": weighted_explainer.explanation_policy,
    }
    blend_metadata_path = _write_json_artifact(
        artifact_dir,
        FULL_WEIGHTED_BLEND_METADATA_FILENAME,
        blend_metadata,
    )

    return {
        "model": selected_model,
        "calibrator": selected_calibrator,
        "metrics": selected_metrics,
        "selected_candidate": selected_candidate,
        "selected_model_family": selected_model_family,
        "selected_model_version": selected_model_version,
        "val_model_roc_auc": selected_val_model_roc_auc,
        "selected_calibrator": selected_calibrator_name,
        "val_policy_calibration_metrics": selected_calibration_metrics,
        "artifacts": selected_runtime_artifacts,
        "fallback_xgboost_artifacts": xgb_candidate_artifacts,
        "blend_metadata_path": blend_metadata_path,
        "candidates": {
            "xgboost_full": xgb_candidate,
            "lightgbm_component_full": lightgbm_component,
            "weighted_blend_full": weighted_candidate,
        },
    }


def _write_meta_blend_experiment_report(artifact_dir: str, report: dict[str, Any]) -> str:
    path = _artifact_path(artifact_dir, META_BLEND_EXPERIMENT_REPORT_FILENAME)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return path


def run_regularized_meta_blend_experiments(
    artifact_dir: str = ARTIFACT_DIR,
    processed_dir: str = DATA_DIR,
    raw_dir: str = "data/raw/",
    full_feature_view: str = DEFAULT_FULL_FEATURE_VIEW,
    regularization_grid: tuple[float, ...] = (0.01, 0.1, 1.0),
) -> dict[str, Any]:
    if not _has_real_training_inputs(processed_dir, raw_dir):
        raise RuntimeError("Meta-blend experiments require real processed splits and raw aggregate tables.")

    bundle = _load_real_bundle(processed_dir, raw_dir)
    with tempfile.TemporaryDirectory(prefix="meta_blend_experiment_build_") as temp_root:
        features = _build_features(bundle, temp_root, processed_dir, full_feature_view=full_feature_view)
        y_train = bundle.train["TARGET"].to_numpy(dtype=int)
        y_val_model = bundle.val_model["TARGET"].to_numpy(dtype=int)
        y_val_policy = bundle.val_policy["TARGET"].to_numpy(dtype=int)
        y_test = bundle.test["TARGET"].to_numpy(dtype=int)

        pos = int(y_train.sum())
        neg = int(len(y_train) - pos)
        scale_pos_weight = float(neg / max(pos, 1))

        xgb_result = _train_best_model_family(
            "xgboost",
            _candidate_model_params(scale_pos_weight),
            features["train_full"],
            y_train,
            features["val_model_full"],
            y_val_model,
        )
        lgbm_result = _train_best_model_family(
            "lightgbm",
            _candidate_lgbm_params(scale_pos_weight),
            features["train_full"],
            y_train,
            features["val_model_full"],
            y_val_model,
        )

        xgb_val_policy_raw = xgb_result["model"].predict_proba(features["val_policy_full"])[:, 1]
        xgb_test_raw = xgb_result["model"].predict_proba(features["test_full"])[:, 1]
        lgbm_val_policy_raw = _predict_lightgbm_raw_pd(lgbm_result["model"], features["val_policy_full"])
        lgbm_test_raw = _predict_lightgbm_raw_pd(lgbm_result["model"], features["test_full"])

        weighted = _select_weighted_average_blend(
            y_val_model,
            xgb_result["val_model_raw_pd"],
            lgbm_result["val_model_raw_pd"],
        )
        weighted_val_policy_raw = weighted["weight_xgb"] * xgb_val_policy_raw + weighted["weight_lgbm"] * lgbm_val_policy_raw
        weighted_test_raw = weighted["weight_xgb"] * xgb_test_raw + weighted["weight_lgbm"] * lgbm_test_raw
        weighted_eval = _evaluate_calibrated_scores(y_val_policy, weighted_val_policy_raw, y_test, weighted_test_raw)

        weighted_candidate = {
            "model_family": "blend",
            "blend_method": "weighted_average",
            "weights": {
                "xgboost": float(weighted["weight_xgb"]),
                "lightgbm": float(weighted["weight_lgbm"]),
            },
            "val_model_roc_auc": float(weighted["val_model_roc_auc"]),
            "test_metrics": weighted_eval["metrics"],
            "selected_calibrator": weighted_eval["selected_calibrator"],
            "val_policy_calibration_metrics": weighted_eval["val_policy_calibration_metrics"],
            "component_selection": {
                "xgboost": {
                    "candidate": xgb_result["selected_candidate"],
                    "params": xgb_result["selected_params"],
                },
                "lightgbm": {
                    "candidate": lgbm_result["selected_candidate"],
                    "params": lgbm_result["selected_params"],
                },
            },
        }

        candidates: dict[str, dict[str, Any]] = {
            "weighted_blend_full": weighted_candidate,
        }
        best_meta_candidate_name: str | None = None
        best_meta_candidate: dict[str, Any] | None = None
        for regularization_c in regularization_grid:
            meta = _fit_logistic_meta_blend(
                y_val_model,
                xgb_result["val_model_raw_pd"],
                lgbm_result["val_model_raw_pd"],
                regularization_c=regularization_c,
            )
            meta_inputs_val_policy = np.column_stack([xgb_val_policy_raw, lgbm_val_policy_raw])
            meta_inputs_test = np.column_stack([xgb_test_raw, lgbm_test_raw])
            meta_val_policy_raw = meta["meta_model"].predict_proba(meta_inputs_val_policy)[:, 1]
            meta_test_raw = meta["meta_model"].predict_proba(meta_inputs_test)[:, 1]
            meta_eval = _evaluate_calibrated_scores(y_val_policy, meta_val_policy_raw, y_test, meta_test_raw)
            candidate_name = f"logistic_meta_blend_full_c_{str(regularization_c).replace('.', '_')}"
            candidate_payload = {
                "model_family": "blend",
                "blend_method": "logistic_meta",
                "regularization_c": float(regularization_c),
                "val_model_roc_auc": float(meta["val_model_roc_auc"]),
                "test_metrics": meta_eval["metrics"],
                "selected_calibrator": meta_eval["selected_calibrator"],
                "val_policy_calibration_metrics": meta_eval["val_policy_calibration_metrics"],
                "meta_coefficients": meta["meta_model"].coef_.tolist(),
                "meta_intercept": meta["meta_model"].intercept_.tolist(),
                "base_inputs": ["xgboost_full_pd", "lightgbm_full_pd"],
            }
            candidates[candidate_name] = candidate_payload
            if best_meta_candidate is None or candidate_payload["val_model_roc_auc"] > best_meta_candidate["val_model_roc_auc"]:
                best_meta_candidate_name = candidate_name
                best_meta_candidate = candidate_payload

        assert best_meta_candidate_name is not None and best_meta_candidate is not None
        best_candidate_name = max(
            candidates.items(),
            key=lambda item: item[1]["val_model_roc_auc"],
        )[0]

        report = {
            "mode": bundle.mode,
            "offline_only": True,
            "processed_manifest_fingerprint": _processed_manifest_fingerprint(processed_dir),
            "full_feature_view": full_feature_view,
            "single_model_runtime_unchanged": True,
            "runtime_baseline_reference": {
                "candidate": "weighted_blend_full",
                "model_version": FULL_WEIGHTED_BLEND_MODEL_VERSION,
            },
            "weighted_blend_baseline": weighted_candidate,
            "best_meta_candidate_by_val_model": best_meta_candidate_name,
            "best_candidate_by_val_model": best_candidate_name,
            "candidates": candidates,
            "notes": [
                "Meta-blend experiments are offline-only and do not replace the default runtime artifact stack.",
                "val_model is used for meta-learner fitting and candidate comparison; val_policy is used for calibration; test is final confirmation only.",
                "Base meta inputs are the FULL XGBoost and FULL LightGBM probabilities.",
            ],
        }

    report_path = _write_meta_blend_experiment_report(artifact_dir, report)
    report["report_path"] = report_path
    return report


def _write_reproducibility_report(artifact_dir: str, report: dict[str, Any]) -> str:
    path = _artifact_path(artifact_dir, REPRODUCIBILITY_REPORT_FILENAME)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return path


def train_models(
    artifact_dir: str = ARTIFACT_DIR,
    processed_dir: str = DATA_DIR,
    raw_dir: str = "data/raw/",
    full_feature_view: str = DEFAULT_FULL_FEATURE_VIEW,
) -> dict[str, Any]:
    os.makedirs(artifact_dir, exist_ok=True)

    if _has_real_training_inputs(processed_dir, raw_dir):
        bundle = _load_real_bundle(processed_dir, raw_dir)
    else:
        bundle = _generate_synthetic_bundle()

    features = _build_features(bundle, artifact_dir, processed_dir, full_feature_view=full_feature_view)
    y_train = bundle.train["TARGET"].to_numpy(dtype=int)
    y_val_model = bundle.val_model["TARGET"].to_numpy(dtype=int)
    y_val_policy = bundle.val_policy["TARGET"].to_numpy(dtype=int)
    y_test = bundle.test["TARGET"].to_numpy(dtype=int)
    processed_manifest_fingerprint = _processed_manifest_fingerprint(processed_dir)

    full_result = _train_full_runtime_candidate(
        features["train_full"],
        y_train,
        features["val_model_full"],
        y_val_model,
        features["val_policy_full"],
        y_val_policy,
        features["test_full"],
        y_test,
        artifact_dir,
        full_feature_view=full_feature_view,
        processed_manifest_fingerprint=processed_manifest_fingerprint,
    )
    reduced_result = _train_tier_model(
        "REDUCED",
        features["train_reduced"],
        y_train,
        features["val_model_reduced"],
        y_val_model,
        features["val_policy_reduced"],
        y_val_policy,
        features["test_reduced"],
        y_test,
        artifact_dir,
    )

    joblib.dump(False, _artifact_path(artifact_dir, FAIRNESS_RESULT_FILENAME))

    report = {
        "mode": bundle.mode,
        "random_state": RANDOM_STATE,
        "model_family": full_result["selected_model_family"],
        "model_version": full_result["selected_model_version"],
        "deployed_model_version": full_result["selected_model_version"],
        "full_model_version": full_result["selected_model_version"],
        "full_xgboost_fallback_model_version": MODEL_VERSIONS["full"],
        "reduced_model_version": MODEL_VERSIONS["reduced"],
        "metrics": full_result["metrics"],
        "reduced_metrics": reduced_result["metrics"],
        "sample_counts": {
            "train": int(len(bundle.train)),
            "val_model": int(len(bundle.val_model)),
            "val_policy": int(len(bundle.val_policy)),
            "test": int(len(bundle.test)),
        },
        "processed_manifest_fingerprint": processed_manifest_fingerprint,
        "tiers": {
            "FULL": {
                "model_family": full_result["selected_model_family"],
                "model_version": full_result["selected_model_version"],
                "feature_view": full_feature_view,
                "feature_count": int(len(features["full_builder"].encoded_columns_)),
                "metrics": full_result["metrics"],
                "selection": {
                    "candidate": full_result["selected_candidate"],
                    "val_model_roc_auc": full_result["val_model_roc_auc"],
                    "calibrator": full_result["selected_calibrator"],
                    "val_policy_calibration_metrics": full_result["val_policy_calibration_metrics"],
                },
                "candidates": full_result["candidates"],
                "artifacts": {
                    "builder": _builder_artifact_path(artifact_dir, "FULL"),
                    "model": _artifact_path(artifact_dir, "full_model.joblib"),
                    "calibrator": _artifact_path(artifact_dir, "full_calibrator.joblib"),
                    "explainer": _artifact_path(artifact_dir, "full_shap_explainer.joblib"),
                    "fallback_xgboost_model": _artifact_path(artifact_dir, f"{FULL_XGBOOST_CANDIDATE_PREFIX}_model.joblib"),
                    "fallback_xgboost_calibrator": _artifact_path(artifact_dir, f"{FULL_XGBOOST_CANDIDATE_PREFIX}_calibrator.joblib"),
                    "fallback_xgboost_explainer": _artifact_path(artifact_dir, f"{FULL_XGBOOST_CANDIDATE_PREFIX}_shap_explainer.joblib"),
                    "weighted_blend_model": _artifact_path(artifact_dir, f"{FULL_WEIGHTED_BLEND_CANDIDATE_PREFIX}_model.joblib"),
                    "weighted_blend_calibrator": _artifact_path(artifact_dir, f"{FULL_WEIGHTED_BLEND_CANDIDATE_PREFIX}_calibrator.joblib"),
                    "weighted_blend_explainer": _artifact_path(artifact_dir, f"{FULL_WEIGHTED_BLEND_CANDIDATE_PREFIX}_shap_explainer.joblib"),
                    "weighted_blend_metadata": full_result["blend_metadata_path"],
                },
            },
            "REDUCED": {
                "model_family": "xgboost",
                "model_version": MODEL_VERSIONS["reduced"],
                "feature_count": int(len(features["reduced_builder"].encoded_columns_)),
                "metrics": reduced_result["metrics"],
                "selection": {
                    "candidate": reduced_result["selected_candidate"],
                    "val_model_roc_auc": reduced_result["val_model_roc_auc"],
                    "params": reduced_result["selected_params"],
                    "calibrator": reduced_result["selected_calibrator"],
                    "val_policy_calibration_metrics": reduced_result["val_policy_calibration_metrics"],
                },
                "artifacts": {
                    "builder": _builder_artifact_path(artifact_dir, "REDUCED"),
                    "model": _artifact_path(artifact_dir, "reduced_model.joblib"),
                    "calibrator": _artifact_path(artifact_dir, "reduced_calibrator.joblib"),
                    "explainer": _artifact_path(artifact_dir, "reduced_shap_explainer.joblib"),
                },
            },
        },
        "artifacts": {
            "fairness_result": _artifact_path(artifact_dir, FAIRNESS_RESULT_FILENAME),
            "reproducibility_report": _artifact_path(artifact_dir, REPRODUCIBILITY_REPORT_FILENAME),
            "full_weighted_blend_metadata": full_result["blend_metadata_path"],
        },
        "blend_evaluation": {
            "evaluated": True,
            "report_path": full_result["blend_metadata_path"],
            "best_candidate": full_result["selected_candidate"],
            "deployed": full_result["selected_candidate"] == "weighted_blend_full",
        },
        "notes": [
            "Synthetic fallback is used when local processed/raw data artifacts are unavailable.",
            "FULL trains an XGBoost fallback candidate and a deployable weighted XGBoost+LightGBM blend candidate under the same processed lineage.",
            "REDUCED remains an XGBoost-only single-model tier in this workflow.",
            "Validation model split is used for runtime-candidate selection; validation policy split is used for probability calibration; test is confirmation only.",
            "Weighted blend explanations use a weighted component Tree SHAP approximation for stable runtime top-feature reasons.",
            "Module 3 writes model_fairness_audit_passed.joblib as a False placeholder. Module 4 owns the final overwrite.",
        ],
    }
    report["report_path"] = _artifact_path(artifact_dir, REPRODUCIBILITY_REPORT_FILENAME)
    _write_reproducibility_report(artifact_dir, report)
    return report

def run_feature_view_ablations(
    processed_dir: str = DATA_DIR,
    raw_dir: str = "data/raw/",
) -> dict[str, Any]:
    if not _has_real_training_inputs(processed_dir, raw_dir):
        raise RuntimeError("Feature-view ablations require real processed splits and raw aggregate tables.")

    view_order = [
        DEFAULT_FULL_FEATURE_VIEW,
        "FULL_NO_CREDIT_CARD",
        "FULL_NO_POS_CASH",
        "FULL_NO_PREVIOUS_APPLICATION",
        "FULL_NO_INSTALLMENTS",
        "FULL_NO_BUREAU",
    ]
    for view_name in view_order:
        resolve_full_feature_view(view_name)

    results: dict[str, Any] = {
        "processed_manifest_fingerprint": _processed_manifest_fingerprint(processed_dir),
        "reduced": {},
        "views": {},
    }

    with tempfile.TemporaryDirectory(prefix="feature_view_ablation_") as temp_root:
        baseline_report = train_models(
            artifact_dir=os.path.join(temp_root, DEFAULT_FULL_FEATURE_VIEW.lower()),
            processed_dir=processed_dir,
            raw_dir=raw_dir,
            full_feature_view=DEFAULT_FULL_FEATURE_VIEW,
        )
        reduced_tier = baseline_report["tiers"]["REDUCED"]
        results["reduced"] = {
            "view": "REDUCED",
            "roc_auc": float(reduced_tier["metrics"]["roc_auc"]),
            "brier_score": float(reduced_tier["metrics"]["brier_score"]),
            "feature_count": int(reduced_tier["feature_count"]),
            "candidate": reduced_tier["selection"]["candidate"],
            "val_model_roc_auc": float(reduced_tier["selection"]["val_model_roc_auc"]),
        }
        full_tier = baseline_report["tiers"]["FULL"]
        results["views"][DEFAULT_FULL_FEATURE_VIEW] = {
            "roc_auc": float(full_tier["metrics"]["roc_auc"]),
            "brier_score": float(full_tier["metrics"]["brier_score"]),
            "feature_count": int(full_tier["feature_count"]),
            "candidate": full_tier["selection"]["candidate"],
            "val_model_roc_auc": float(full_tier["selection"]["val_model_roc_auc"]),
        }

        for view_name in view_order[1:]:
            report = train_models(
                artifact_dir=os.path.join(temp_root, view_name.lower()),
                processed_dir=processed_dir,
                raw_dir=raw_dir,
                full_feature_view=view_name,
            )
            tier = report["tiers"]["FULL"]
            results["views"][view_name] = {
                "roc_auc": float(tier["metrics"]["roc_auc"]),
                "brier_score": float(tier["metrics"]["brier_score"]),
                "feature_count": int(tier["feature_count"]),
                "candidate": tier["selection"]["candidate"],
                "val_model_roc_auc": float(tier["selection"]["val_model_roc_auc"]),
            }

    return results


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
    print(_format_metric_line("REDUCED", report["reduced_metrics"]))
    print(f"Report written to: {report['report_path']}")

