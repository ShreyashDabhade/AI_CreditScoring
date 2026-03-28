"""SQLite insertion utilities for applicant records.

Provides:
- insert_applicant(applicant_record: dict, db_path: str) -> bool
- batch_insert_applicants(records: list[dict], db_path: str) -> int

This module uses `db_schema` to create the `applicants` table if missing and
performs parameterized inserts (INSERT OR REPLACE).
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
from typing import Any, Dict, List

import numpy as np
import pandas as pd

import db_schema

logger = logging.getLogger(__name__)

# Fixed column names expected in schema (order matters in db_schema.FIXED_COLUMNS)
_FIXED_COLS = [name for name, _ in db_schema.FIXED_COLUMNS]
_CHATBOT_REQUIRED_COLUMNS = {
    "SK_ID_CURR",
    "prediction",
    "credit_score",
    "risk_band",
    "probability",
    "top_features",
    "explanation_text",
}


def _derive_feature_names_from_record(record: Dict) -> List[str]:
    """Return ordered list of dynamic feature names from applicant record.

    We exclude the fixed columns from the record keys. The function preserves
    insertion order of the dict; if not available, falls back to sorted keys.
    """
    if not isinstance(record, dict):
        raise ValueError("record must be a dict")
    dynamic = [k for k in record.keys() if k not in _FIXED_COLS]
    if dynamic:
        return dynamic
    # fallback
    return sorted([k for k in record.keys() if k not in _FIXED_COLS])


def insert_applicant(applicant_record: Dict, db_path: str) -> bool:
    """Insert a single applicant record into SQLite.

    - Creates schema if missing using `db_schema.create_schema`.
    - Uses `INSERT OR REPLACE` with parameterized values.
    - Returns True on success, False on failure (logs error).
    """
    try:
        feature_names = _derive_feature_names_from_record(applicant_record)
        # ensure table exists with these feature columns
        db_schema.create_schema(db_path, feature_names)

        # get sanitized column list in db order
        cols = db_schema.get_column_names(feature_names)

        # map dynamic sanitized names back to original feature keys
        fixed_count = len(_FIXED_COLS)
        dynamic_sanitized = cols[fixed_count:]
        mapping = {san: orig for san, orig in zip(dynamic_sanitized, feature_names)}

        # build values tuple aligned to cols order
        values = []
        for col in cols:
            if col in _FIXED_COLS:
                values.append(applicant_record.get(col))
            else:
                orig = mapping.get(col)
                values.append(applicant_record.get(orig))

        placeholders = ",".join(["?" for _ in cols])
        col_list = ",".join(cols)
        stmt = f"INSERT OR REPLACE INTO applicants ({col_list}) VALUES ({placeholders})"

        with sqlite3.connect(db_path) as conn:
            cur = conn.cursor()
            cur.execute(stmt, tuple(values))
            conn.commit()
        return True
    except Exception as e:
        logger.exception("Failed to insert applicant %s: %s", applicant_record.get("applicant_id"), e)
        return False


def batch_insert_applicants(records: List[Dict], db_path: str) -> int:
    """Batch insert multiple applicant records using executemany.

    Returns the count of successful inserts (attempted records length on success,
    or number inserted before failure). Errors are logged and function returns
    the number successfully inserted.
    """
    if not records:
        return 0

    try:
        # derive feature names from first record (assume consistent schema)
        feature_names = _derive_feature_names_from_record(records[0])
        db_schema.create_schema(db_path, feature_names)
        cols = db_schema.get_column_names(feature_names)
        fixed_count = len(_FIXED_COLS)
        dynamic_sanitized = cols[fixed_count:]
        mapping = {san: orig for san, orig in zip(dynamic_sanitized, feature_names)}

        placeholders = ",".join(["?" for _ in cols])
        col_list = ",".join(cols)
        stmt = f"INSERT OR REPLACE INTO applicants ({col_list}) VALUES ({placeholders})"

        values_list = []
        for rec in records:
            vals = []
            for col in cols:
                if col in _FIXED_COLS:
                    vals.append(rec.get(col))
                else:
                    orig = mapping.get(col)
                    vals.append(rec.get(orig))
            values_list.append(tuple(vals))

        with sqlite3.connect(db_path) as conn:
            cur = conn.cursor()
            cur.executemany(stmt, values_list)
            conn.commit()

        return len(values_list)
    except Exception as e:
        logger.exception("Batch insert failed: %s", e)
        # attempt best-effort: try inserting one-by-one to count successes
        success = 0
        try:
            with sqlite3.connect(db_path) as conn:
                cur = conn.cursor()
                for rec in records:
                    try:
                        feature_names = _derive_feature_names_from_record(rec)
                        db_schema.create_schema(db_path, feature_names)
                        cols = db_schema.get_column_names(feature_names)
                        fixed_count = len(_FIXED_COLS)
                        dynamic_sanitized = cols[fixed_count:]
                        mapping = {san: orig for san, orig in zip(dynamic_sanitized, feature_names)}

                        vals = []
                        for col in cols:
                            if col in _FIXED_COLS:
                                vals.append(rec.get(col))
                            else:
                                orig = mapping.get(col)
                                vals.append(rec.get(orig))
                        placeholders = ",".join(["?" for _ in cols])
                        col_list = ",".join(cols)
                        cur.execute(f"INSERT OR REPLACE INTO applicants ({col_list}) VALUES ({placeholders})", tuple(vals))
                        success += 1
                    except Exception:
                        logger.exception("Failed to insert one record with applicant_id=%s", rec.get("applicant_id"))
                conn.commit()
        except Exception:
            logger.exception("Fallback one-by-one insert also failed")
        return success


def _safe_json_dumps(value: Any) -> str:
    """Serialize arbitrary values into a JSON string."""
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        return str(value)


def _prediction_label(probability: float, approve_threshold: float) -> str:
    """Convert probability of default into chatbot decision labels."""
    return "APPROVE" if probability < approve_threshold else "REJECT"


def _risk_band_label(probability: float) -> str:
    """Map probability of default to a coarse risk band."""
    if probability < 0.30:
        return "LOW"
    if probability < 0.60:
        return "MEDIUM"
    return "HIGH"


def _credit_score(probability: float) -> int:
    """Convert probability of default into a 300-850 style score."""
    return int(max(300, min(850, round((1.0 - probability) * 850.0))))


def _join_reasons(reasons: list[str]) -> str:
    """Join explanation phrases into a short readable list."""
    cleaned = [reason.strip() for reason in reasons if reason and reason.strip()]
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0]
    if len(cleaned) == 2:
        return f"{cleaned[0]} and {cleaned[1]}"
    return ", ".join(cleaned[:-1]) + f", and {cleaned[-1]}"


def _build_explanation_text(prediction: str, probability: float, reasons: list[str]) -> str:
    """Create a human-readable explanation narrative for one applicant."""
    probability_pct = f"{probability * 100:.1f}%"
    if prediction == "APPROVE":
        opening = (
            f"The application was approved with an estimated default probability of {probability_pct}."
        )
    else:
        opening = (
            f"The application was rejected with an estimated default probability of {probability_pct}."
        )

    if not reasons:
        return opening

    return f"{opening} The main factors were {_join_reasons(reasons[:3])}."


def _normalize_batch_shap_output(raw_shap: Any, row_count: int, feature_count: int) -> np.ndarray:
    """Normalize explainer output into a 2D SHAP matrix."""
    try:
        import shap  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        shap = None

    if shap is not None and isinstance(raw_shap, shap.Explanation):
        raw_shap = raw_shap.values

    if isinstance(raw_shap, list):
        if not raw_shap:
            raise RuntimeError("Explainer returned an empty SHAP list")
        raw_shap = raw_shap[1] if len(raw_shap) > 1 else raw_shap[0]

    values = np.asarray(raw_shap, dtype=float)
    if values.ndim == 3:
        class_index = 1 if values.shape[-1] > 1 else 0
        values = values[..., class_index]

    if values.ndim == 1:
        if row_count != 1 or values.shape[0] != feature_count:
            raise RuntimeError("Unexpected 1D SHAP output")
        return values.reshape(1, -1)

    if values.ndim == 2:
        if values.shape == (row_count, feature_count):
            return values
        if values.shape == (feature_count, row_count):
            return values.T

    raise RuntimeError("Explainer returned SHAP values with an unexpected shape")


def _extract_reason_rows(features: pd.DataFrame, explainer: Any, chunk_size: int = 256) -> list[list[str]]:
    """Derive top reason phrases for each applicant row from SHAP output."""
    from src.explainability import top_5_explanations_from_shap

    reason_rows: list[list[str]] = []
    for start in range(0, len(features), chunk_size):
        stop = start + chunk_size
        chunk = features.iloc[start:stop]
        try:
            raw_input = chunk.to_numpy(dtype=float, copy=False)
            if callable(explainer):
                raw_shap = explainer(raw_input)
            elif hasattr(explainer, "shap_values") and callable(explainer.shap_values):
                raw_shap = explainer.shap_values(raw_input)
            else:
                raise RuntimeError("Explainer is not callable and has no shap_values method")

            shap_matrix = _normalize_batch_shap_output(raw_shap, len(chunk), chunk.shape[1])
        except Exception as exc:
            logger.warning("Falling back to generic explanations because SHAP failed: %s", exc)
            shap_matrix = np.zeros((len(chunk), chunk.shape[1]), dtype=float)

        for row_index in range(len(chunk)):
            shap_series = pd.Series(shap_matrix[row_index], index=features.columns, dtype=float)
            top_items = top_5_explanations_from_shap(shap_series)
            reasons: list[str] = []
            for item in top_items[:3]:
                reason = str(item.get("reason") or item.get("feature") or "").strip()
                if reason and reason not in reasons:
                    reasons.append(reason)
            reason_rows.append(reasons)

    return reason_rows


def _load_scoring_source(processed_dir: str, raw_dir: str) -> pd.DataFrame:
    """Load the full application test cohort used to populate the chatbot DB."""
    raw_test_path = os.path.join(raw_dir, "application_test.csv")
    processed_adv_path = os.path.join(processed_dir, "app_test_adv.pkl")

    raw_df = pd.read_csv(raw_test_path)
    if "SK_ID_CURR" not in raw_df.columns:
        raise RuntimeError("application_test.csv is missing SK_ID_CURR")

    if os.path.exists(processed_adv_path):
        adv_obj = pd.read_pickle(processed_adv_path)
        if isinstance(adv_obj, dict) and {"adv_train", "adv_val"}.issubset(adv_obj):
            combined = pd.concat([adv_obj["adv_train"], adv_obj["adv_val"]], ignore_index=True)
            if "TARGET" in combined.columns:
                candidate = combined[combined["TARGET"].isna()].copy()
            else:
                candidate = combined.copy()

            if not candidate.empty and "SK_ID_CURR" in candidate.columns:
                merged = candidate.merge(
                    raw_df,
                    on="SK_ID_CURR",
                    how="left",
                    suffixes=("", "__raw"),
                )
                for column_name in raw_df.columns:
                    if column_name == "SK_ID_CURR":
                        continue
                    raw_column = f"{column_name}__raw"
                    if raw_column not in merged.columns:
                        continue
                    if column_name in merged.columns:
                        merged[column_name] = merged[column_name].where(
                            merged[column_name].notna(),
                            merged[raw_column],
                        )
                    else:
                        merged[column_name] = merged[raw_column]
                    merged = merged.drop(columns=[raw_column])
                raw_df = merged

    if "AMT_INCOME_TOTAL_CAPPED" not in raw_df.columns and "AMT_INCOME_TOTAL" in raw_df.columns:
        raw_df["AMT_INCOME_TOTAL_CAPPED"] = pd.to_numeric(
            raw_df["AMT_INCOME_TOTAL"],
            errors="coerce",
        )

    if "AGE_YEARS" not in raw_df.columns and "DAYS_BIRTH" in raw_df.columns:
        age_years = pd.to_numeric(raw_df["DAYS_BIRTH"], errors="coerce").abs() // 365
        raw_df["AGE_YEARS"] = age_years.astype("Int64")

    return raw_df.loc[:, ~raw_df.columns.duplicated()].copy()


def get_applicants_db_summary(db_path: str) -> dict[str, Any]:
    """Return row counts, sample IDs, and schema details for the applicants table."""
    summary: dict[str, Any] = {
        "db_path": db_path,
        "has_table": False,
        "columns": [],
        "total_applicants": 0,
        "sample_ids": [],
        "prediction_breakdown": {},
        "contains_100038": False,
    }

    if not os.path.exists(db_path):
        return summary

    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='applicants'"
        )
        if cur.fetchone() is None:
            return summary

        summary["has_table"] = True
        cur.execute("PRAGMA table_info(applicants)")
        columns = [str(row[1]) for row in cur.fetchall()]
        summary["columns"] = columns

        cur.execute("SELECT COUNT(*) FROM applicants")
        summary["total_applicants"] = int(cur.fetchone()[0])

        if "SK_ID_CURR" in columns:
            cur.execute("SELECT SK_ID_CURR FROM applicants ORDER BY SK_ID_CURR LIMIT 5")
            summary["sample_ids"] = [int(row[0]) for row in cur.fetchall()]
            cur.execute("SELECT 1 FROM applicants WHERE SK_ID_CURR = 100038 LIMIT 1")
            summary["contains_100038"] = cur.fetchone() is not None
        elif "applicant_id" in columns:
            cur.execute("SELECT applicant_id FROM applicants ORDER BY applicant_id LIMIT 5")
            summary["sample_ids"] = [int(row[0]) for row in cur.fetchall()]

        if "prediction" in columns:
            cur.execute("SELECT prediction, COUNT(*) FROM applicants GROUP BY prediction")
            summary["prediction_breakdown"] = {
                str(label): int(count) for label, count in cur.fetchall()
            }

    return summary


def rebuild_chatbot_database(
    db_path: str,
    runtime: Any,
    *,
    processed_dir: str = "data/processed",
    raw_dir: str = "data/raw",
    chunk_size: int = 256,
) -> dict[str, Any]:
    """Rebuild the chatbot applicants database from the scored full test cohort."""
    from configs.config import APPROVE_THRESHOLD

    source_df = _load_scoring_source(processed_dir=processed_dir, raw_dir=raw_dir)
    if source_df.empty:
        raise RuntimeError("No applicant rows were available to populate the chatbot DB")

    builder = getattr(runtime, "full_builder", None)
    model = getattr(runtime, "full_model", None)
    calibrator = getattr(runtime, "full_calibrator", None)
    explainer = getattr(runtime, "full_shap_explainer", None)
    if builder is None or model is None or calibrator is None or explainer is None:
        raise RuntimeError("Runtime does not expose the FULL scoring components required for DB rebuild")

    if not hasattr(builder, "transform") or not callable(builder.transform):
        raise RuntimeError("FULL builder does not expose transform(...)")

    features = builder.transform(source_df)
    if not isinstance(features, pd.DataFrame) or len(features) != len(source_df):
        raise RuntimeError("FULL builder returned an invalid feature matrix")

    probabilities_matrix = np.asarray(model.predict_proba(features), dtype=float)
    if probabilities_matrix.ndim != 2 or probabilities_matrix.shape[0] != len(source_df) or probabilities_matrix.shape[1] < 2:
        raise RuntimeError("FULL model predict_proba returned an unexpected shape")
    raw_pd = probabilities_matrix[:, 1]

    try:
        calibrated = np.asarray(calibrator.predict(raw_pd), dtype=float).reshape(-1)
        if calibrated.size != raw_pd.size:
            raise RuntimeError("Calibrator returned the wrong number of probabilities")
    except Exception:
        calibrated = np.asarray(
            [
                float(np.asarray(calibrator.predict(np.array([value], dtype=float)), dtype=float).reshape(-1)[0])
                for value in raw_pd
            ],
            dtype=float,
        )

    calibrated = np.clip(calibrated, 0.0, 1.0)
    reason_rows = _extract_reason_rows(features, explainer, chunk_size=chunk_size)

    scored_df = source_df.copy()
    scored_df["SK_ID_CURR"] = pd.to_numeric(scored_df["SK_ID_CURR"], errors="coerce").astype("Int64")
    scored_df["probability"] = calibrated
    scored_df["prediction"] = [
        _prediction_label(float(probability), float(APPROVE_THRESHOLD))
        for probability in calibrated
    ]
    scored_df["credit_score"] = [_credit_score(float(probability)) for probability in calibrated]
    scored_df["risk_band"] = [_risk_band_label(float(probability)) for probability in calibrated]
    scored_df["top_features"] = [
        ", ".join(reasons[:3]) if reasons else "Not available"
        for reasons in reason_rows
    ]
    scored_df["explanation_text"] = [
        _build_explanation_text(prediction, float(probability), reasons)
        for prediction, probability, reasons in zip(
            scored_df["prediction"],
            calibrated,
            reason_rows,
        )
    ]

    if "AMT_INCOME_TOTAL_CAPPED" not in scored_df.columns and "AMT_INCOME_TOTAL" in scored_df.columns:
        scored_df["AMT_INCOME_TOTAL_CAPPED"] = pd.to_numeric(
            scored_df["AMT_INCOME_TOTAL"],
            errors="coerce",
        )

    if "AGE_YEARS" not in scored_df.columns and "DAYS_BIRTH" in scored_df.columns:
        age_years = pd.to_numeric(scored_df["DAYS_BIRTH"], errors="coerce").abs() // 365
        scored_df["AGE_YEARS"] = age_years.astype("Int64")

    preferred_order = [
        "SK_ID_CURR",
        "prediction",
        "credit_score",
        "risk_band",
        "probability",
        "AMT_INCOME_TOTAL_CAPPED",
        "AMT_CREDIT",
        "AMT_ANNUITY",
        "AGE_YEARS",
        "DAYS_EMPLOYED",
        "CODE_GENDER",
        "NAME_EDUCATION_TYPE",
        "CNT_FAM_MEMBERS",
        "top_features",
        "explanation_text",
    ]
    remaining_columns = [
        column_name
        for column_name in scored_df.columns
        if column_name not in preferred_order
    ]
    scored_df = scored_df[[column_name for column_name in preferred_order if column_name in scored_df.columns] + remaining_columns]

    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        scored_df.to_sql("applicants", conn, if_exists="replace", index=False)
        cur = conn.cursor()
        cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_applicants_sk_id_curr ON applicants(SK_ID_CURR)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_applicants_prediction ON applicants(prediction)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_applicants_risk_band ON applicants(risk_band)")
        conn.commit()

    summary = get_applicants_db_summary(db_path)
    summary["source_rows"] = int(len(source_df))
    return summary


def ensure_chatbot_database(
    db_path: str,
    runtime: Any,
    *,
    processed_dir: str = "data/processed",
    raw_dir: str = "data/raw",
    auto_rebuild: bool = True,
) -> dict[str, Any]:
    """Ensure the chatbot DB uses the modern scored-applicant schema."""
    summary = get_applicants_db_summary(db_path)
    has_required_schema = _CHATBOT_REQUIRED_COLUMNS.issubset(set(summary.get("columns", [])))
    looks_populated = int(summary.get("total_applicants", 0)) > 0
    if has_required_schema and looks_populated:
        return summary

    if not auto_rebuild:
        logger.warning("Chatbot applicants DB is invalid but auto rebuild is disabled: %s", summary)
        return summary

    logger.info("Rebuilding chatbot applicants DB at %s", db_path)
    return rebuild_chatbot_database(
        db_path,
        runtime,
        processed_dir=processed_dir,
        raw_dir=raw_dir,
    )
