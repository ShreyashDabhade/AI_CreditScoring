"""Authoritative model training and reproducibility entrypoint."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import platform
from dataclasses import dataclass
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".mplconfig"))

import joblib
import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd
matplotlib.use("Agg")
import shap
import sklearn
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.builder_artifacts import (
    BuilderValidationError,
    DEFAULT_FULL_BUILDER_PATH,
    DEFAULT_REDUCED_BUILDER_PATH,
    builder_manifest_path,
    current_git_commit,
    dataframe_schema_hash,
    dataset_fingerprint,
    load_builder_artifact,
    load_processed_artifact_manifest,
    processed_manifest_path,
    quarantine_existing_paths,
)
from src.feature_engineering import (
    AGGREGATE_CONTRACT_VERSION,
    ALL_AGGREGATE_FEATURE_COLS,
    APPLICATION_REQUIRED_INPUT_COLS,
    FEATURE_ENGINEERING_VERSION,
    _agg_bureau,
    _agg_credit_card,
    _agg_installments,
    _agg_pos_cash,
    _agg_previous,
    _select_application_frame,
    build_full,
    build_reduced,
    fit_full_builder,
    fit_reduced_builder,
    safe_left_merge_one_to_one,
)
from src.models.artifact_types import ProbabilityCalibrator, SerializableShapExplainer

SEED = 42
N_JOBS = 1
POLICY_APPROVE_THRESHOLD = 0.15
POLICY_REVIEW_THRESHOLD = 0.35

FULL_MODEL_ARTIFACT_PATH = "artifacts/full_model.joblib"
FULL_CALIBRATOR_ARTIFACT_PATH = "artifacts/full_calibrator.joblib"
FULL_SHAP_EXPLAINER_ARTIFACT_PATH = "artifacts/full_shap_explainer.joblib"
REPRODUCIBILITY_REPORT_PATH = "artifacts/reproducibility_report.json"

PROCESSED_SPLIT_NAMES = ["train", "val_model", "val_policy", "test"]
PROCESSED_DATA_RECOVERY_NOTE = (
    "Metrics after processed-data recovery are under the corrected "
    "forward proxy-time regime and supersede all prior split-based metrics."
)


@dataclass
class CandidateResult:
    model_name: str
    tier: str
    estimator: Any
    feature_count: int
    train_auc: float
    val_model_auc: float
    val_policy_auc: float
    test_auc: float
    raw_scores: dict[str, np.ndarray]


def decision_from_pd(pd_value: float) -> str:
    if pd_value < POLICY_APPROVE_THRESHOLD:
        return "APPROVE"
    if pd_value < POLICY_REVIEW_THRESHOLD:
        return "REVIEW"
    return "DECLINE"


def load_artifacts(artifact_dir: str = "artifacts") -> dict[str, Any]:
    return {
        "full_model": joblib.load(os.path.join(artifact_dir, "full_model.joblib")),
        "full_calibrator": joblib.load(os.path.join(artifact_dir, "full_calibrator.joblib")),
        "full_shap_explainer": joblib.load(os.path.join(artifact_dir, "full_shap_explainer.joblib")),
    }


def _load_processed_split(split_name: str, data_processed_dir: str) -> pd.DataFrame:
    path = os.path.join(data_processed_dir, f"{split_name}.pkl")
    with open(path, "rb") as handle:
        df = pickle.load(handle)
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{path} did not load as a pandas DataFrame")
    return df


def _sorted_split(df: pd.DataFrame) -> pd.DataFrame:
    if "SK_ID_CURR" in df.columns:
        return df.sort_values("SK_ID_CURR", kind="mergesort").reset_index(drop=True)
    return df.reset_index(drop=True)


def _load_processed_splits(data_processed_dir: str) -> dict[str, pd.DataFrame]:
    return {
        split_name: _sorted_split(_load_processed_split(split_name, data_processed_dir))
        for split_name in PROCESSED_SPLIT_NAMES
    }


def _assert_lineage_equal(actual: Any, expected: Any, message: str) -> None:
    if actual != expected:
        raise BuilderValidationError(f"{message}: expected {expected!r}, found {actual!r}")


def _validate_processed_split_against_manifest(
    split_name: str,
    df: pd.DataFrame,
    manifest: dict[str, Any],
) -> None:
    summary = manifest["split_summary"][split_name]
    _assert_lineage_equal(int(len(df)), int(summary["rows"]), f"{split_name} row count mismatch")
    _assert_lineage_equal(
        int(df["TARGET"].sum()),
        int(summary["positive_count"]),
        f"{split_name} positive count mismatch",
    )
    if not np.isclose(float(df["TARGET"].mean()), float(summary["target_rate"])):
        raise BuilderValidationError(
            f"{split_name} target rate mismatch: expected {summary['target_rate']!r}, "
            f"found {float(df['TARGET'].mean())!r}"
        )
    if not np.isclose(
        float(df["DAYS_ID_PUBLISH"].abs().mean()),
        float(summary["mean_abs_days_id_publish"]),
    ):
        raise BuilderValidationError(f"{split_name} mean_abs_days_id_publish mismatch")
    if not np.isclose(
        float(df["DAYS_REGISTRATION"].abs().mean()),
        float(summary["mean_abs_days_registration"]),
    ):
        raise BuilderValidationError(f"{split_name} mean_abs_days_registration mismatch")
    if not np.isclose(
        float(df["DAYS_EMPLOYED_ANOM"].mean()),
        float(summary["days_employed_anom_rate"]),
    ):
        raise BuilderValidationError(f"{split_name} days_employed_anom_rate mismatch")

    _assert_lineage_equal(
        dataset_fingerprint(df),
        manifest["split_fingerprints"][split_name],
        f"{split_name} fingerprint mismatch",
    )
    _assert_lineage_equal(
        dataframe_schema_hash(df),
        manifest["split_schema_hashes"][split_name],
        f"{split_name} schema hash mismatch",
    )


def _validate_processed_manifest_lineage(
    splits: dict[str, pd.DataFrame],
    manifest: dict[str, Any],
) -> None:
    for split_name, df in splits.items():
        _validate_processed_split_against_manifest(split_name, df, manifest)

    total_rows = sum(len(df) for df in splits.values())
    _assert_lineage_equal(
        total_rows,
        int(manifest["application_train_cleaned_rows"]),
        "Scoring split row preservation mismatch",
    )

    for split_name, df in splits.items():
        dupes = int(df["SK_ID_CURR"].duplicated().sum())
        _assert_lineage_equal(dupes, 0, f"{split_name} duplicate SK_ID_CURR count mismatch")

    combined_ids = pd.concat([df["SK_ID_CURR"] for df in splits.values()], ignore_index=True)
    _assert_lineage_equal(
        int(combined_ids.duplicated().sum()),
        0,
        "Cross-split duplicate SK_ID_CURR count mismatch",
    )


def _load_and_validate_income_cap(
    data_processed_dir: str,
    manifest: dict[str, Any],
    train_df: pd.DataFrame,
) -> float:
    income_cap_path = os.path.join(data_processed_dir, "income_cap.joblib")
    income_cap = float(joblib.load(income_cap_path))
    if not np.isclose(income_cap, float(manifest["income_cap"])):
        raise BuilderValidationError("income_cap.joblib does not match processed manifest")
    if not np.isclose(income_cap, float(train_df["AMT_INCOME_TOTAL"].quantile(0.99))):
        raise BuilderValidationError("income_cap.joblib does not match train split 99th percentile")
    return income_cap


def _load_and_validate_adversarial_artifact(
    data_processed_dir: str,
    manifest: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    adv_path = os.path.join(data_processed_dir, "app_test_adv.pkl")
    with open(adv_path, "rb") as handle:
        adv = pickle.load(handle)

    if not isinstance(adv, dict) or set(adv.keys()) != {"adv_train", "adv_val"}:
        raise BuilderValidationError("app_test_adv.pkl must be a dict with adv_train and adv_val")

    adv_summary = manifest["adversarial_summary"]
    frames = {key: value for key, value in adv.items()}
    reference_non_label_columns = None

    for split_name, expected_rows_key in [("adv_train", "adv_train_rows"), ("adv_val", "adv_val_rows")]:
        df = frames[split_name]
        if not isinstance(df, pd.DataFrame):
            raise BuilderValidationError(f"{split_name} did not load as a DataFrame")
        _assert_lineage_equal(
            int(len(df)),
            int(adv_summary[expected_rows_key]),
            f"{split_name} row count mismatch",
        )
        if "ADV_LABEL" not in df.columns:
            raise BuilderValidationError(f"{split_name} is missing ADV_LABEL")
        if "TARGET" in df.columns:
            raise BuilderValidationError(f"{split_name} must not contain TARGET")
        for required_column in ["AMT_INCOME_TOTAL_CAPPED", "DAYS_EMPLOYED_ANOM"]:
            if required_column not in df.columns:
                raise BuilderValidationError(f"{split_name} is missing {required_column}")
        if int(df["SK_ID_CURR"].duplicated().sum()) != 0:
            raise BuilderValidationError(f"{split_name} contains duplicate SK_ID_CURR values")
        non_label_columns = [column for column in df.columns if column != "ADV_LABEL"]
        if reference_non_label_columns is None:
            reference_non_label_columns = non_label_columns
        elif non_label_columns != reference_non_label_columns:
            raise BuilderValidationError(
                "adv_train and adv_val must share identical non-label column order"
            )
        expected_capped = np.minimum(df["AMT_INCOME_TOTAL"], float(adv_summary["income_cap_used"]))
        if int((~np.isclose(df["AMT_INCOME_TOTAL_CAPPED"], expected_capped, equal_nan=True)).sum()) != 0:
            raise BuilderValidationError(f"{split_name} AMT_INCOME_TOTAL_CAPPED does not match manifest cap")

    _assert_lineage_equal(
        dataframe_schema_hash(frames["adv_train"]),
        adv_summary["schema_hash"],
        "Adversarial schema hash mismatch",
    )
    _assert_lineage_equal(
        bool(adv_summary["diagnostic_only"]),
        True,
        "Adversarial diagnostic_only flag mismatch",
    )
    return frames


def _load_processed_lineage(
    data_processed_dir: str,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any], float, dict[str, pd.DataFrame]]:
    manifest = load_processed_artifact_manifest(processed_manifest_path(data_processed_dir))
    splits = _load_processed_splits(data_processed_dir)
    _validate_processed_manifest_lineage(splits, manifest)
    income_cap = _load_and_validate_income_cap(data_processed_dir, manifest, splits["train"])
    adversarial_frames = _load_and_validate_adversarial_artifact(data_processed_dir, manifest)
    return splits, manifest, income_cap, adversarial_frames


def _model_metrics(y_true: np.ndarray, proba: np.ndarray) -> float:
    return float(roc_auc_score(y_true, proba))


def _train_logistic_regression(X_train: pd.DataFrame, y_train: np.ndarray) -> LogisticRegression:
    model = LogisticRegression(
        max_iter=4000,
        random_state=SEED,
        solver="lbfgs",
        C=1.0,
    )
    model.fit(X_train.values, y_train)
    return model


def _train_xgboost(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val_model: pd.DataFrame,
    y_val_model: np.ndarray,
) -> xgb.XGBClassifier:
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="auc",
        random_state=SEED,
        n_jobs=N_JOBS,
        tree_method="hist",
        use_label_encoder=False,
    )
    model.fit(
        X_train.values,
        y_train,
        eval_set=[(X_val_model.values, y_val_model)],
        verbose=False,
    )
    return model


def _train_lightgbm(X_train: pd.DataFrame, y_train: np.ndarray) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        random_state=SEED,
        n_jobs=N_JOBS,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )
    model.fit(X_train.values, y_train)
    return model


def _evaluate_candidate(
    *,
    model_name: str,
    tier: str,
    estimator: Any,
    feature_sets: dict[str, pd.DataFrame],
    labels: dict[str, np.ndarray],
) -> CandidateResult:
    raw_scores = {
        split_name: estimator.predict_proba(features.values)[:, 1]
        for split_name, features in feature_sets.items()
    }
    return CandidateResult(
        model_name=model_name,
        tier=tier,
        estimator=estimator,
        feature_count=int(feature_sets["train"].shape[1]),
        train_auc=_model_metrics(labels["train"], raw_scores["train"]),
        val_model_auc=_model_metrics(labels["val_model"], raw_scores["val_model"]),
        val_policy_auc=_model_metrics(labels["val_policy"], raw_scores["val_policy"]),
        test_auc=_model_metrics(labels["test"], raw_scores["test"]),
        raw_scores=raw_scores,
    )


def _fit_calibrator(raw_scores: np.ndarray, y_true: np.ndarray, model_name: str) -> ProbabilityCalibrator:
    regressor = IsotonicRegression(out_of_bounds="clip")
    regressor.fit(raw_scores, y_true)
    return ProbabilityCalibrator(
        method="isotonic_regression",
        fitted_on_split="val_policy",
        model_name=model_name,
        regressor=regressor,
    )


def _build_shap_explainer(model: Any, X_background: pd.DataFrame) -> SerializableShapExplainer:
    background = X_background.sample(
        n=min(256, len(X_background)),
        random_state=SEED,
    ).to_numpy(dtype=np.float32)
    return SerializableShapExplainer(
        model=model,
        background=background,
        feature_names=list(X_background.columns),
    )


def _quarantine_manifest_summary(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "builder_tier": manifest["builder_tier"],
        "fit_split_name": manifest["fit_split_name"],
        "fit_row_count": manifest["fit_row_count"],
        "dataset_fingerprint": manifest["dataset_fingerprint"],
        "feature_engineering_version": manifest["feature_engineering_version"],
        "git_commit": manifest["git_commit"],
        "processed_manifest_fingerprint": manifest.get("processed_manifest_fingerprint"),
        "aggregate_contract_version": manifest["aggregate_contract_version"],
        "encoded_column_schema_hash": manifest["encoded_column_schema_hash"],
        "pre_model_column_schema_hash": manifest["pre_model_column_schema_hash"],
        "encoded_column_count": manifest["encoded_column_count"],
        "rare_map_sizes": manifest["rare_map_sizes"],
        "categorical_columns": manifest["categorical_columns"],
        "rare_map_schema_hash": manifest["rare_map_schema_hash"],
        "fit_timestamp": manifest["fit_timestamp"],
    }


def _split_summary(splits: dict[str, pd.DataFrame]) -> dict[str, dict[str, float | int]]:
    return {
        split_name: {
            "rows": int(len(df)),
            "positive_count": int(df["TARGET"].sum()),
            "target_rate": float(df["TARGET"].mean()),
        }
        for split_name, df in splits.items()
    }


def _duplicate_summary(splits: dict[str, pd.DataFrame]) -> dict[str, int]:
    combined_ids = pd.concat([df["SK_ID_CURR"] for df in splits.values()], ignore_index=True)
    return {
        **{
            f"{split_name}_duplicate_sk_id_curr": int(df["SK_ID_CURR"].duplicated().sum())
            for split_name, df in splits.items()
        },
        "cross_split_duplicate_sk_id_curr": int(combined_ids.duplicated().sum()),
    }


def _build_flattened_full_inputs(
    splits: dict[str, pd.DataFrame],
    raw_dir: str,
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, int]]]:
    aggregate_tables = [
        ("bureau", _agg_bureau(raw_dir)),
        ("previous_application", _agg_previous(raw_dir)),
        ("installments", _agg_installments(raw_dir)),
        ("pos_cash", _agg_pos_cash(raw_dir)),
        ("credit_card", _agg_credit_card(raw_dir)),
    ]

    flattened_inputs: dict[str, pd.DataFrame] = {}
    merge_integrity: dict[str, dict[str, int]] = {}

    for split_name, split_df in splits.items():
        base = _select_application_frame(split_df, require_sk_id_curr=True)
        merged = base.copy()
        for feature_name, agg_df in aggregate_tables:
            merged = safe_left_merge_one_to_one(
                merged,
                agg_df,
                key="SK_ID_CURR",
                feat_name=feature_name,
            )
        flattened_inputs[split_name] = merged
        merge_integrity[split_name] = {
            "input_rows": int(len(split_df)),
            "output_rows": int(len(merged)),
            "flattened_full_rows_match": int(len(split_df) == len(merged)),
        }

    return flattened_inputs, merge_integrity


def _prepare_feature_sets(
    splits: dict[str, pd.DataFrame],
    flattened_full_inputs: dict[str, pd.DataFrame],
    full_builder: Any,
    reduced_builder: Any,
    raw_dir: str,
) -> dict[tuple[str, str], dict[str, pd.DataFrame]]:
    feature_sets: dict[tuple[str, str], dict[str, pd.DataFrame]] = {}

    feature_sets[("REDUCED", "linear")] = {
        split_name: build_reduced(split_df, reduced_builder, for_linear_model=True)
        for split_name, split_df in splits.items()
    }
    feature_sets[("REDUCED", "tree")] = {
        split_name: build_reduced(split_df, reduced_builder, for_linear_model=False)
        for split_name, split_df in splits.items()
    }
    feature_sets[("FULL", "linear")] = {
        split_name: build_full(flattened_full_inputs[split_name], full_builder, for_linear_model=True)
        for split_name in splits
    }
    feature_sets[("FULL", "tree")] = {
        split_name: build_full(flattened_full_inputs[split_name], full_builder, for_linear_model=False)
        for split_name in splits
    }
    return feature_sets


def _candidate_metric_table(results: list[CandidateResult]) -> list[dict[str, Any]]:
    ordered = sorted(results, key=lambda item: (item.tier, item.model_name))
    return [
        {
            "model_name": result.model_name,
            "tier": result.tier,
            "feature_count": result.feature_count,
            "metric_kind": "uncalibrated_probability_auc",
            "train_auc": result.train_auc,
            "val_model_auc": result.val_model_auc,
            "val_policy_auc": result.val_policy_auc,
            "test_auc": result.test_auc,
        }
        for result in ordered
    ]


def _select_official_champion(results: list[CandidateResult]) -> CandidateResult:
    full_results = [result for result in results if result.tier == "FULL"]
    if not full_results:
        raise RuntimeError("Champion selection requires FULL-tier candidates")
    return max(full_results, key=lambda result: (result.val_model_auc, result.test_auc))


def _adversarial_validation_summary(
    *,
    adversarial_frames: dict[str, pd.DataFrame],
    processed_manifest_fingerprint: str,
) -> dict[str, Any]:
    adv_train_df = adversarial_frames["adv_train"]
    adv_val_df = adversarial_frames["adv_val"]
    adv_builder = fit_reduced_builder(adv_train_df)
    X_train = build_reduced(adv_train_df, adv_builder, for_linear_model=True)
    X_val = build_reduced(adv_val_df, adv_builder, for_linear_model=True)
    y_train = adv_train_df["ADV_LABEL"].to_numpy()
    y_val = adv_val_df["ADV_LABEL"].to_numpy()
    adv_model = _train_logistic_regression(X_train, y_train)
    train_auc = _model_metrics(y_train, adv_model.predict_proba(X_train.values)[:, 1])
    val_auc = _model_metrics(y_val, adv_model.predict_proba(X_val.values)[:, 1])
    return {
        "metric_kind": "uncalibrated_probability_auc",
        "seed": SEED,
        "n_jobs": N_JOBS,
        "train_rows": int(len(adv_train_df)),
        "val_rows": int(len(adv_val_df)),
        "train_positive_rate": float(y_train.mean()),
        "val_positive_rate": float(y_val.mean()),
        "feature_count": int(X_train.shape[1]),
        "builder_artifact_path": None,
        "builder_manifest_path": None,
        "processed_manifest_fingerprint": processed_manifest_fingerprint,
        "train_auc": train_auc,
        "val_auc": val_auc,
    }


def _one_family_ablation_table(
    splits: dict[str, pd.DataFrame],
    reduced_builder: Any,
    raw_dir: str,
) -> list[dict[str, Any]]:
    labels = {split_name: df["TARGET"].to_numpy() for split_name, df in splits.items()}
    reduced_tree = {
        split_name: build_reduced(split_df, reduced_builder, for_linear_model=False)
        for split_name, split_df in splits.items()
    }
    aggregate_families = {
        "bureau": _agg_bureau(raw_dir),
        "previous_application": _agg_previous(raw_dir),
        "installments": _agg_installments(raw_dir),
        "pos_cash": _agg_pos_cash(raw_dir),
        "credit_card": _agg_credit_card(raw_dir),
    }

    results: list[dict[str, Any]] = []
    for family_name, family_df in aggregate_families.items():
        ablation_features: dict[str, pd.DataFrame] = {}
        added_cols = [col for col in family_df.columns if col != "SK_ID_CURR"]
        for split_name, split_df in splits.items():
            merged = split_df[["SK_ID_CURR"]].merge(
                family_df,
                on="SK_ID_CURR",
                how="left",
                validate="one_to_one",
            )
            features = reduced_tree[split_name].copy()
            for column in added_cols:
                features[column] = merged[column].values
            ablation_features[split_name] = features

        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="auc",
            random_state=SEED,
            n_jobs=N_JOBS,
            tree_method="hist",
            use_label_encoder=False,
        )
        model.fit(
            ablation_features["train"].values,
            labels["train"],
            eval_set=[(ablation_features["val_model"].values, labels["val_model"])],
            verbose=False,
        )
        val_auc = _model_metrics(
            labels["val_model"],
            model.predict_proba(ablation_features["val_model"].values)[:, 1],
        )
        test_auc = _model_metrics(
            labels["test"],
            model.predict_proba(ablation_features["test"].values)[:, 1],
        )
        results.append(
            {
                "family": family_name,
                "feature_count": int(ablation_features["train"].shape[1]),
                "val_model_auc": val_auc,
                "test_auc": test_auc,
            }
        )
    return results


def _environment_metadata() -> dict[str, str]:
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "scikit_learn_version": sklearn.__version__,
        "xgboost_version": xgb.__version__,
        "lightgbm_version": lgb.__version__,
        "joblib_version": joblib.__version__,
        "shap_version": shap.__version__,
    }


def run_training(
    *,
    data_processed_dir: str = "data/processed",
    data_raw_dir: str = "data/raw",
    artifact_dir: str = "artifacts",
    require_git_match: bool = False,
) -> dict[str, Any]:
    os.makedirs(artifact_dir, exist_ok=True)
    full_builder_path = os.path.join(artifact_dir, os.path.basename(DEFAULT_FULL_BUILDER_PATH))
    reduced_builder_path = os.path.join(artifact_dir, os.path.basename(DEFAULT_REDUCED_BUILDER_PATH))
    full_model_path = os.path.join(artifact_dir, "full_model.joblib")
    full_calibrator_path = os.path.join(artifact_dir, "full_calibrator.joblib")
    full_shap_explainer_path = os.path.join(artifact_dir, "full_shap_explainer.joblib")
    reproducibility_report_path = os.path.join(artifact_dir, "reproducibility_report.json")
    processed_manifest_file = processed_manifest_path(data_processed_dir)

    splits, processed_manifest, _, adversarial_frames = _load_processed_lineage(data_processed_dir)
    processed_manifest_fp = processed_manifest["processed_manifest_fingerprint"]
    labels = {split_name: df["TARGET"].to_numpy() for split_name, df in splits.items()}

    quarantine_existing_paths(
        [
            full_builder_path,
            builder_manifest_path(full_builder_path),
            reduced_builder_path,
            builder_manifest_path(reduced_builder_path),
            full_model_path,
            full_calibrator_path,
            full_shap_explainer_path,
            reproducibility_report_path,
        ],
        os.path.join(artifact_dir, "quarantine", "training_outputs"),
    )

    reduced_builder = fit_reduced_builder(
        splits["train"],
        artifact_path=reduced_builder_path,
        fit_split_name="train",
        strict_artifact_validation=True,
        processed_manifest_fingerprint=processed_manifest_fp,
    )
    full_builder = fit_full_builder(
        splits["train"],
        raw_dir=data_raw_dir,
        artifact_path=full_builder_path,
        fit_split_name="train",
        strict_artifact_validation=True,
        processed_manifest_fingerprint=processed_manifest_fp,
    )

    reduced_builder, reduced_manifest, reduced_warnings = load_builder_artifact(
        reduced_builder_path,
        fit_df=splits["train"],
        expected_tier="REDUCED",
        fit_split_name="train",
        feature_engineering_version=FEATURE_ENGINEERING_VERSION,
        aggregate_contract_version=AGGREGATE_CONTRACT_VERSION,
        expected_aggregate_feature_count=0,
        require_git_match=require_git_match,
        expected_processed_manifest_fingerprint=processed_manifest_fp,
    )
    full_builder, full_manifest, full_warnings = load_builder_artifact(
        full_builder_path,
        fit_df=splits["train"],
        expected_tier="FULL",
        fit_split_name="train",
        feature_engineering_version=FEATURE_ENGINEERING_VERSION,
        aggregate_contract_version=AGGREGATE_CONTRACT_VERSION,
        expected_aggregate_feature_count=len(ALL_AGGREGATE_FEATURE_COLS),
        require_git_match=require_git_match,
        expected_processed_manifest_fingerprint=processed_manifest_fp,
    )

    flattened_full_inputs, merge_integrity = _build_flattened_full_inputs(splits, data_raw_dir)
    feature_sets = _prepare_feature_sets(
        splits,
        flattened_full_inputs,
        full_builder,
        reduced_builder,
        data_raw_dir,
    )
    feature_counts = {
        "REDUCED_linear": int(feature_sets[("REDUCED", "linear")]["train"].shape[1]),
        "REDUCED_tree": int(feature_sets[("REDUCED", "tree")]["train"].shape[1]),
        "FULL_linear": int(feature_sets[("FULL", "linear")]["train"].shape[1]),
        "FULL_tree": int(feature_sets[("FULL", "tree")]["train"].shape[1]),
    }

    results: list[CandidateResult] = []
    results.append(
        _evaluate_candidate(
            model_name="logistic_regression",
            tier="REDUCED",
            estimator=_train_logistic_regression(
                feature_sets[("REDUCED", "linear")]["train"],
                labels["train"],
            ),
            feature_sets=feature_sets[("REDUCED", "linear")],
            labels=labels,
        )
    )
    results.append(
        _evaluate_candidate(
            model_name="xgboost",
            tier="REDUCED",
            estimator=_train_xgboost(
                feature_sets[("REDUCED", "tree")]["train"],
                labels["train"],
                feature_sets[("REDUCED", "tree")]["val_model"],
                labels["val_model"],
            ),
            feature_sets=feature_sets[("REDUCED", "tree")],
            labels=labels,
        )
    )
    results.append(
        _evaluate_candidate(
            model_name="lightgbm",
            tier="REDUCED",
            estimator=_train_lightgbm(
                feature_sets[("REDUCED", "tree")]["train"],
                labels["train"],
            ),
            feature_sets=feature_sets[("REDUCED", "tree")],
            labels=labels,
        )
    )
    results.append(
        _evaluate_candidate(
            model_name="logistic_regression",
            tier="FULL",
            estimator=_train_logistic_regression(
                feature_sets[("FULL", "linear")]["train"],
                labels["train"],
            ),
            feature_sets=feature_sets[("FULL", "linear")],
            labels=labels,
        )
    )
    results.append(
        _evaluate_candidate(
            model_name="xgboost",
            tier="FULL",
            estimator=_train_xgboost(
                feature_sets[("FULL", "tree")]["train"],
                labels["train"],
                feature_sets[("FULL", "tree")]["val_model"],
                labels["val_model"],
            ),
            feature_sets=feature_sets[("FULL", "tree")],
            labels=labels,
        )
    )
    results.append(
        _evaluate_candidate(
            model_name="lightgbm",
            tier="FULL",
            estimator=_train_lightgbm(
                feature_sets[("FULL", "tree")]["train"],
                labels["train"],
            ),
            feature_sets=feature_sets[("FULL", "tree")],
            labels=labels,
        )
    )

    champion = _select_official_champion(results)
    champion_feature_mode = "linear" if champion.model_name == "logistic_regression" else "tree"
    champion_feature_sets = feature_sets[(champion.tier, champion_feature_mode)]
    calibrator = _fit_calibrator(
        champion.raw_scores["val_policy"],
        labels["val_policy"],
        champion.model_name,
    )
    calibrated_val_policy = calibrator.predict(champion.raw_scores["val_policy"])
    calibrated_test = calibrator.predict(champion.raw_scores["test"])
    calibrated_metrics = {
        "metric_kind": "calibrated_probability_auc",
        "val_policy_auc": _model_metrics(labels["val_policy"], calibrated_val_policy),
        "test_auc": _model_metrics(labels["test"], calibrated_test),
    }

    shap_explainer = _build_shap_explainer(
        champion.estimator,
        champion_feature_sets["train"],
    )

    joblib.dump(champion.estimator, full_model_path)
    joblib.dump(calibrator, full_calibrator_path)
    joblib.dump(shap_explainer, full_shap_explainer_path)

    adversarial_summary = _adversarial_validation_summary(
        adversarial_frames=adversarial_frames,
        processed_manifest_fingerprint=processed_manifest_fp,
    )
    ablation_results = _one_family_ablation_table(splits, reduced_builder, data_raw_dir)

    reduced_xgb = next(
        result for result in results if result.tier == "REDUCED" and result.model_name == "xgboost"
    )
    reopen_aggregate_hypothesis = bool(
        champion.val_model_auc < (reduced_xgb.val_model_auc - 0.005)
        or any(
            entry["val_model_auc"] < (reduced_xgb.val_model_auc - 0.005)
            and entry["test_auc"] < (reduced_xgb.test_auc - 0.005)
            for entry in ablation_results
        )
    )

    report = {
        "seed": SEED,
        "n_jobs": N_JOBS,
        "feature_engineering_version": FEATURE_ENGINEERING_VERSION,
        "aggregate_contract_version": AGGREGATE_CONTRACT_VERSION,
        "git_commit": current_git_commit(),
        "processed_lineage": {
            "processed_manifest_path": processed_manifest_file,
            "processed_manifest_fingerprint": processed_manifest_fp,
        },
        "diagnostic_notes": [
            PROCESSED_DATA_RECOVERY_NOTE,
        ],
        "builder_artifacts": {
            "full": {
                "builder_path": full_builder_path,
                "manifest_path": builder_manifest_path(full_builder_path),
                "manifest": _quarantine_manifest_summary(full_manifest),
                "warnings": full_warnings,
            },
            "reduced": {
                "builder_path": reduced_builder_path,
                "manifest_path": builder_manifest_path(reduced_builder_path),
                "manifest": _quarantine_manifest_summary(reduced_manifest),
                "warnings": reduced_warnings,
            },
        },
        "model_artifacts": {
            "full_model_path": full_model_path,
            "full_calibrator_path": full_calibrator_path,
            "full_shap_explainer_path": full_shap_explainer_path,
        },
        "split_summary": processed_manifest["split_summary"],
        "duplicate_summary": _duplicate_summary(splits),
        "merge_integrity": merge_integrity,
        "feature_counts": feature_counts,
        "metrics_table": _candidate_metric_table(results),
        "champion_selection": {
            "official_champion_model": champion.model_name,
            "official_champion_tier": champion.tier,
            "selection_metric": "uncalibrated_probability_auc",
            "selection_split": "val_model",
            "val_model_auc": champion.val_model_auc,
            "test_confirmation_auc": champion.test_auc,
            "calibrated_metrics": calibrated_metrics,
        },
        "adversarial_validation": adversarial_summary,
        "one_family_ablations": ablation_results,
        "reopen_aggregate_hypothesis": reopen_aggregate_hypothesis,
        "environment": _environment_metadata(),
        "reproducibility_report_path": reproducibility_report_path,
    }

    with open(reproducibility_report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)

    return report


def _print_summary(report: dict[str, Any]) -> None:
    print("=" * 72)
    print("Committed Training Summary")
    print("=" * 72)
    print("Split sizes and target rates:")
    for split_name, summary in report["split_summary"].items():
        print(
            f"  {split_name:10s} rows={summary['rows']:>7,d} "
            f"positives={summary['positive_count']:>6,d} "
            f"rate={summary['target_rate']:.4f}"
        )
    print("\nMetrics (probability AUC):")
    for row in report["metrics_table"]:
        print(
            f"  {row['model_name']:>20s} {row['tier']:>7s} "
            f"train={row['train_auc']:.4f} "
            f"val_model={row['val_model_auc']:.4f} "
            f"test={row['test_auc']:.4f}"
        )
    champion = report["champion_selection"]
    print(
        "\nOfficial champion: "
        f"{champion['official_champion_model']} ({champion['official_champion_tier']}) "
        f"via {champion['selection_split']} {champion['selection_metric']}="
        f"{champion['val_model_auc']:.4f}"
    )
    print(
        "Calibrated confirmation: "
        f"val_policy={champion['calibrated_metrics']['val_policy_auc']:.4f} "
        f"test={champion['calibrated_metrics']['test_auc']:.4f}"
    )
    print(
        "Adversarial validation: "
        f"val_auc={report['adversarial_validation']['val_auc']:.4f}"
    )
    for note in report.get("diagnostic_notes", []):
        print(f"Note: {note}")
    print(f"Reproducibility report: {report['reproducibility_report_path']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-processed-dir", default="data/processed")
    parser.add_argument("--data-raw-dir", default="data/raw")
    parser.add_argument("--artifact-dir", default="artifacts")
    parser.add_argument(
        "--require-git-match",
        action="store_true",
        help="Hard-fail builder validation when manifests were built on a different git commit.",
    )
    args = parser.parse_args()

    report = run_training(
        data_processed_dir=args.data_processed_dir,
        data_raw_dir=args.data_raw_dir,
        artifact_dir=args.artifact_dir,
        require_git_match=args.require_git_match,
    )
    _print_summary(report)


if __name__ == "__main__":
    main()
