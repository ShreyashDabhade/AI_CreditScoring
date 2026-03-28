"""Example evaluation script that uses the JSON logger.

- Simulates predictions and ground truth (small dummy dataset)
- Computes basic metrics (accuracy, precision, recall, f1, roc_auc)
- Calls functions from src.logging.json_logger to write a single JSON run file

This script is minimal, has no extra dependencies, and is safe to run.
"""
import os
import random
from typing import List
import sys
from pathlib import Path

# Ensure project root is on sys.path so `src` package can be imported when running the script directly
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from src.logging.json_logger import (
    build_metadata,
    build_prediction_row,
    write_run_json,
)

import pandas as pd
import joblib
import numpy as np
from sklearn import metrics as skm


def _rankdata(a: List[float]) -> List[float]:
    # simple rank implementation with average ranks for ties
    indexed = list(enumerate(a))
    sorted_idx = sorted(indexed, key=lambda x: x[1])
    ranks = [0.0] * len(a)
    i = 0
    while i < len(sorted_idx):
        j = i
        total_rank = 0
        while j < len(sorted_idx) and sorted_idx[j][1] == sorted_idx[i][1]:
            total_rank += j + 1
            j += 1
        avg_rank = total_rank / (j - i)
        for k in range(i, j):
            orig_idx = sorted_idx[k][0]
            ranks[orig_idx] = avg_rank
        i = j
    return ranks


def roc_auc_score_manual(y_true: List[int], y_score: List[float]) -> float:
    # Mann-Whitney U statistic based AUC
    n = len(y_true)
    if n == 0:
        return 0.0
    ranks = _rankdata(y_score)
    pos_ranks_sum = 0.0
    n_pos = 0
    n_neg = 0
    for yt, r in zip(y_true, ranks):
        if yt == 1:
            pos_ranks_sum += r
            n_pos += 1
        else:
            n_neg += 1
    if n_pos == 0 or n_neg == 0:
        return 0.0
    auc = (pos_ranks_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def compute_basic_metrics(y_true, y_pred, y_score):
    n = len(y_true)
    tp = fp = tn = fn = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == 1 and yp == 1:
            tp += 1
        elif yt == 0 and yp == 1:
            fp += 1
        elif yt == 0 and yp == 0:
            tn += 1
        elif yt == 1 and yp == 0:
            fn += 1
    accuracy = (tp + tn) / n if n else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
    roc_auc = roc_auc_score_manual(y_true, y_score)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
    }


def _series_to_pydict(s):
    """Convert a pandas Series to a JSON-friendly dict (convert numpy types)."""
    out = {}
    for k, v in s.items():
        if v is None:
            out[k] = None
            continue
        if hasattr(v, "item") and not isinstance(v, (str, bool)):
            try:
                out[k] = v.item()
                continue
            except Exception:
                pass
        if isinstance(v, (list, tuple)):
            out[k] = list(v)
            continue
        out[k] = v
    return out


def main():
    random.seed(0)

    # Preferred resources
    raw_test_path = "data/raw/application_test.csv"
    processed_test_path = "data/processed/test.pkl"
    artifact_dir = "artifacts"
    preferred_models = [
        "full_model.joblib",
        "full_weighted_blend_model.joblib",
        "reduced_model.joblib",
    ]

    # Load model artifact
    model = None
    model_path = None
    for name in preferred_models:
        candidate = os.path.join(artifact_dir, name)
        if os.path.exists(candidate):
            try:
                model = joblib.load(candidate)
                model_path = candidate
                print(f"Loaded model artifact: {candidate}")
                break
            except Exception as exc:
                print(f"Failed to load model {candidate}:", exc)

    if model is None:
        raise RuntimeError("No model artifact found in artifacts/; cannot run evaluation")

    # Load processed test input (builder expects processed application columns)
    if not os.path.exists(processed_test_path):
        raise RuntimeError(f"Processed test file not found: {processed_test_path}")
    processed_df = pd.read_pickle(processed_test_path)
    print(f"Loaded processed test data: {processed_test_path} (n={len(processed_df)})")

    # Extract applicant IDs before transformation for tracking
    if "SK_ID_CURR" in processed_df.columns:
        applicant_ids = processed_df["SK_ID_CURR"].tolist()
    else:
        applicant_ids = list(processed_df.index.astype(str))

    # Load FrozenFeatureBuilder and transform processed data
    from src.builder_artifacts import load_builder

    builder = load_builder(
        tier="FULL", artifact_dir=artifact_dir, processed_dir="data/processed/", strict_artifacts=False
    )
    # For FULL builders, the transform requires processed application columns; provide raw_dir so
    # builder can load auxiliary raw tables for aggregates when needed.
    X = builder.transform(processed_df, raw_dir="data/raw/")
    source_df_for_features = processed_df
    print("Transformed processed test data using FrozenFeatureBuilder")
    print(f"Transformed data -> feature matrix (n_rows={X.shape[0]}, n_cols={X.shape[1]})")

    # Predict probabilities using model.predict_proba (do NOT call model.predict)
    try:
        probs = model.predict_proba(X)
        if probs.ndim == 2 and probs.shape[1] >= 2:
            y_pred_proba = probs[:, 1].tolist()
        else:
            y_pred_proba = np.asarray(probs).ravel().tolist()
    except Exception as exc:
        raise RuntimeError("Model predict_proba failed") from exc

    decision_threshold = 0.5
    y_pred = [1 if p >= decision_threshold else 0 for p in y_pred_proba]

    # Attempt to load ground truth from processed test if present and align by SK_ID_CURR
    y_true = [None] * len(y_pred)
    if os.path.exists(processed_test_path):
        try:
            processed_df = pd.read_pickle(processed_test_path)
            # prefer SK_ID_CURR key for join
            key = None
            for k in ("SK_ID_CURR", "applicant_id"):
                if k in source_df_for_features.columns and k in processed_df.columns:
                    key = k
                    break
            if key is not None and "TARGET" in processed_df.columns:
                merged = source_df_for_features[[key]].merge(processed_df[[key, "TARGET"]], on=key, how="left")
                y_true = merged["TARGET"].fillna(-1).apply(lambda v: int(v) if v != -1 else None).tolist()
                print("Aligned y_true from processed test data")
        except Exception:
            pass

    # selected features for presentation: pick from the source dataframe used for features
    preferred_select = [
        "AMT_INCOME_TOTAL",
        "DAYS_BIRTH",
        "CNT_CHILDREN",
        "AMT_CREDIT",
    ]
    selected_features_list = []
    for _, row in source_df_for_features.iterrows():
        small = {k: row[k] for k in preferred_select if k in source_df_for_features.columns}
        if not small:
            numeric_cols = source_df_for_features.select_dtypes(include=[np.number]).columns.tolist()
            small = {k: row[k] for k in numeric_cols[:3]}
        selected_features_list.append(small)

    # model name/version for metadata
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    model_version = getattr(model, "__version__", "v1")

    # Align applicant ids and selected features to transformed feature matrix length (X)
    if "SK_ID_CURR" in source_df_for_features.columns and len(source_df_for_features) == len(X):
        applicant_ids = source_df_for_features["SK_ID_CURR"].tolist()
        selected_src_df = source_df_for_features
    else:
        # Fall back to X index as identifiers when lengths differ
        applicant_ids = [str(i) for i in X.index.tolist()]
        selected_src_df = None

    # compute metrics using sklearn for correctness where y_true available
    valid_idx = [i for i, yt in enumerate(y_true) if yt is not None]
    if valid_idx:
        y_true_valid = [y_true[i] for i in valid_idx]
        y_pred_valid = [y_pred[i] for i in valid_idx]
        y_score_valid = [y_pred_proba[i] for i in valid_idx]
        metrics = {
            "accuracy": float(skm.accuracy_score(y_true_valid, y_pred_valid)),
            "precision": float(skm.precision_score(y_true_valid, y_pred_valid, zero_division=0)),
            "recall": float(skm.recall_score(y_true_valid, y_pred_valid, zero_division=0)),
            "f1": float(skm.f1_score(y_true_valid, y_pred_valid, zero_division=0)),
            "roc_auc": float(skm.roc_auc_score(y_true_valid, y_score_valid)),
        }
    else:
        metrics = compute_basic_metrics(y_true, y_pred, y_pred_proba)
    
    metadata = build_metadata(
        model_name=model_name,
        model_version=model_version,
        dataset_split=("test"),
        n_samples=len(y_pred),
        roc_auc=metrics["roc_auc"],
        accuracy=metrics["accuracy"],
        precision=metrics["precision"],
        recall=metrics["recall"],
        f1=metrics["f1"],
        decision_threshold=decision_threshold,
    )

    predictions = []
    # Use applicant_ids (aligned to X) as row identifiers for logging
    ids_for_rows = applicant_ids

    # Ensure all iterables align to the number of predictions
    n_pred = len(y_pred_proba)
    ids_for_rows = ids_for_rows[:n_pred]
    selected_features_list = selected_features_list[:n_pred]
    y_pred = y_pred[:n_pred]
    y_true = y_true[:n_pred]

    # Build prediction rows and attach full transformed features as `raw_features`.
    for i in range(n_pred):
        aid = ids_for_rows[i]
        feats = selected_features_list[i] if i < len(selected_features_list) else {}
        proba = y_pred_proba[i]
        predc = y_pred[i]
        truth = y_true[i] if i < len(y_true) else None

        # Extract corresponding transformed feature row from X when possible
        full_row = {}
        try:
                if hasattr(X, "iloc") and len(X) > i:
                    full_row = _series_to_pydict(X.iloc[i])
                    # Ensure SK_ID_CURR is present in raw_features for traceability
                    try:
                        full_row["SK_ID_CURR"] = applicant_ids[i]
                    except Exception:
                        pass
        except Exception:
            full_row = {}

        row = build_prediction_row(
            applicant_id=aid,
            selected_features=feats,
            pred_proba=proba,
            pred_class=predc,
            true_class=truth,
            decision_threshold=decision_threshold,
            top_features=None,
        )
        # Attach raw_features (full model input features) for CSV output and LLM helpers
        row["raw_features"] = full_row
        predictions.append(row)

    from src.logging.json_logger import write_llm_and_csv
    print("Before metadata creation")

    run_obj = {"metadata": metadata, "predictions": predictions}
    print(" metadata created")

    json_path, csv_path = write_llm_and_csv(run_obj, logs_dir="logs")

    print("LLM JSON written to:", json_path)
    print("CSV written to:", csv_path)


if __name__ == "__main__":
    main()
