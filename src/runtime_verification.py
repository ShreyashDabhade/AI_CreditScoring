"""Shared verification and lineage helpers for runtime artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

EXPECTED_PROCESSED_SPLITS: tuple[str, ...] = ("train", "val_model", "val_policy", "test")


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"Object of type {type(value)} is not JSON serializable")


def stable_json_dumps(payload: Any) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    )


def sha256_json(payload: Any) -> str:
    return hashlib.sha256(stable_json_dumps(payload).encode("utf-8")).hexdigest()


def column_sequence_hash(columns: Iterable[str]) -> str:
    return sha256_json(list(columns))


def rare_map_schema_hash(rare_map: Mapping[str, set[str]]) -> str:
    return sha256_json({key: sorted(value) for key, value in sorted(rare_map.items())})


def dataframe_schema_hash(df: pd.DataFrame) -> str:
    payload = [{"name": str(col), "dtype": str(df[col].dtype)} for col in df.columns]
    return sha256_json(payload)


def dataframe_fingerprint(
    df: pd.DataFrame,
    columns: Iterable[str] | None = None,
) -> str:
    if columns is None:
        preferred = [
            "SK_ID_CURR",
            "TARGET",
            "ADV_LABEL",
            "DAYS_ID_PUBLISH",
            "DAYS_REGISTRATION",
        ]
        selected = [column for column in preferred if column in df.columns]
        if not selected:
            selected = list(df.columns[: min(len(df.columns), 10)])
    else:
        selected = [column for column in columns if column in df.columns]
        if not selected:
            raise ValueError("No requested fingerprint columns are present in the DataFrame")

    subset = df[selected].copy()
    hashed = pd.util.hash_pandas_object(subset, index=False).to_numpy(dtype=np.uint64)
    payload = {
        "rows": int(len(df)),
        "columns": selected,
        "schema_hash": dataframe_schema_hash(subset),
        "hash_sum_mod64": int(hashed.sum(dtype=np.uint64)),
    }
    return sha256_json(payload)


def build_split_summary(df: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {"rows": int(len(df))}

    if "TARGET" in df.columns:
        target = pd.to_numeric(df["TARGET"], errors="coerce")
        summary["positive_count"] = int(target.fillna(0).sum())
        summary["target_rate"] = float(target.mean()) if len(target) else 0.0

    if "DAYS_EMPLOYED_ANOM" in df.columns:
        anom = pd.to_numeric(df["DAYS_EMPLOYED_ANOM"], errors="coerce")
        summary["days_employed_anom_rate"] = float(anom.fillna(0).mean()) if len(anom) else 0.0

    if "DAYS_ID_PUBLISH" in df.columns:
        recency = pd.to_numeric(df["DAYS_ID_PUBLISH"], errors="coerce").abs()
        summary["mean_abs_days_id_publish"] = float(recency.mean()) if len(recency) else 0.0

    if "DAYS_REGISTRATION" in df.columns:
        recency = pd.to_numeric(df["DAYS_REGISTRATION"], errors="coerce").abs()
        summary["mean_abs_days_registration"] = float(recency.mean()) if len(recency) else 0.0

    return summary


def safe_git_commit(cwd: str | None = None) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None

    commit = result.stdout.strip()
    return commit or None


def load_json_object(path: str, label: str) -> dict[str, Any]:
    if not os.path.exists(path):
        raise RuntimeError(f"Missing required {label}: {path}")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception as exc:
        raise RuntimeError(f"Failed to read {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} must contain a JSON object: {path}")
    return payload


def validate_processed_splits(splits: Mapping[str, pd.DataFrame]) -> dict[str, Any]:
    missing = [name for name in EXPECTED_PROCESSED_SPLITS if name not in splits]
    if missing:
        raise ValueError(f"Missing processed splits for verification: {missing}")

    duplicate_summary: dict[str, int] = {}
    split_schema_hashes: dict[str, str] = {}
    split_fingerprints: dict[str, str] = {}
    split_summary: dict[str, dict[str, Any]] = {}

    for split_name in EXPECTED_PROCESSED_SPLITS:
        df = splits[split_name]
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"{split_name} must be a pandas DataFrame")
        if "SK_ID_CURR" not in df.columns:
            raise ValueError(f"{split_name} is missing SK_ID_CURR")

        dupes = int(df["SK_ID_CURR"].duplicated().sum())
        duplicate_summary[f"{split_name}_duplicate_sk_id_curr"] = dupes
        if dupes:
            raise ValueError(f"{split_name} contains duplicate SK_ID_CURR values")

        split_schema_hashes[split_name] = dataframe_schema_hash(df)
        split_fingerprints[split_name] = dataframe_fingerprint(df)
        split_summary[split_name] = build_split_summary(df)

    all_ids = pd.concat(
        [
            splits[split_name][["SK_ID_CURR"]].assign(__split=split_name)
            for split_name in EXPECTED_PROCESSED_SPLITS
        ],
        ignore_index=True,
    )
    cross_split_dupes = int(all_ids["SK_ID_CURR"].duplicated().sum())
    duplicate_summary["cross_split_duplicate_sk_id_curr"] = cross_split_dupes
    if cross_split_dupes:
        raise ValueError("Processed splits reuse SK_ID_CURR across partitions")

    unique_schema_hashes = set(split_schema_hashes.values())
    if len(unique_schema_hashes) != 1:
        raise ValueError("Processed splits do not share an identical schema")

    for left, right in zip(EXPECTED_PROCESSED_SPLITS, EXPECTED_PROCESSED_SPLITS[1:]):
        left_publish = pd.to_numeric(splits[left]["DAYS_ID_PUBLISH"], errors="coerce").abs().dropna()
        right_publish = pd.to_numeric(splits[right]["DAYS_ID_PUBLISH"], errors="coerce").abs().dropna()
        if not left_publish.empty and not right_publish.empty:
            if float(left_publish.min()) < float(right_publish.max()):
                raise ValueError(
                    "Proxy-time split ordering violated between "
                    f"{left} and {right} on DAYS_ID_PUBLISH"
                )

        left_registration = pd.to_numeric(splits[left]["DAYS_REGISTRATION"], errors="coerce").abs().dropna()
        right_registration = pd.to_numeric(splits[right]["DAYS_REGISTRATION"], errors="coerce").abs().dropna()
        if not left_registration.empty and not right_registration.empty:
            if float(left_registration.mean()) < float(right_registration.mean()):
                raise ValueError(
                    "Proxy-time split ordering violated between "
                    f"{left} and {right} on DAYS_REGISTRATION mean"
                )

    return {
        "duplicate_summary": duplicate_summary,
        "split_schema_hashes": split_schema_hashes,
        "split_fingerprints": split_fingerprints,
        "split_summary": split_summary,
    }


def validate_transformed_frame(
    df: pd.DataFrame,
    *,
    expected_rows: int | None = None,
    expected_columns: Iterable[str] | None = None,
    forbidden_columns: Iterable[str] = (),
    fail_on_constant_columns: bool = False,
) -> dict[str, Any]:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Builder transform must return a pandas DataFrame")

    if expected_rows is not None and len(df) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, got {len(df)}")

    non_numeric = [column for column in df.columns if not pd.api.types.is_numeric_dtype(df[column])]
    if non_numeric:
        raise ValueError(f"Transformed frame contains non-numeric columns: {non_numeric}")

    if expected_columns is not None and list(df.columns) != list(expected_columns):
        raise ValueError("Transformed frame column order does not match the frozen builder contract")

    blocked = [column for column in df.columns if column in set(forbidden_columns)]
    if blocked:
        raise ValueError(f"Forbidden columns leaked into transformed frame: {blocked}")

    all_null_columns = [column for column in df.columns if df[column].isna().all()]
    if all_null_columns:
        raise ValueError(f"Transformed frame contains all-null columns: {all_null_columns}")

    constant_columns: list[str] = []
    if fail_on_constant_columns:
        constant_columns = [column for column in df.columns if df[column].nunique(dropna=False) <= 1]
        if constant_columns:
            raise ValueError(f"Transformed frame contains constant columns: {constant_columns}")

    return {
        "rows": int(len(df)),
        "feature_count": int(df.shape[1]),
        "schema_hash": dataframe_schema_hash(df),
        "all_null_columns": all_null_columns,
        "constant_columns": constant_columns,
    }


def validate_builder_artifact(
    builder: Any,
    *,
    tier: str,
    manifest: Mapping[str, Any] | None,
    processed_manifest: Mapping[str, Any] | None = None,
    strict: bool,
) -> dict[str, Any]:
    from src.feature_engineering import FrozenFeatureBuilder

    normalized_tier = tier.upper()
    if not isinstance(builder, FrozenFeatureBuilder):
        raise TypeError(f"{normalized_tier} builder must be a FrozenFeatureBuilder")
    if builder.tier.upper() != normalized_tier:
        raise ValueError(
            f"{normalized_tier} builder tier mismatch: got {builder.tier!r}"
        )

    warnings: list[str] = []
    if manifest is None:
        if strict:
            raise RuntimeError(f"Missing required {normalized_tier} builder manifest")
        warnings.append(f"{normalized_tier} builder manifest missing; strict lineage checks skipped")
        return {"warnings": warnings}

    manifest_tier = str(manifest.get("builder_tier", "")).upper()
    if manifest_tier != normalized_tier:
        raise RuntimeError(
            f"{normalized_tier} builder manifest tier mismatch: {manifest_tier!r}"
        )

    encoded_count = int(manifest.get("encoded_column_count", -1))
    if encoded_count != len(builder.encoded_columns_):
        raise RuntimeError(
            f"{normalized_tier} builder encoded column count mismatch"
        )

    expected_encoded_hash = column_sequence_hash(builder.encoded_columns_)
    if manifest.get("encoded_column_schema_hash") != expected_encoded_hash:
        raise RuntimeError(
            f"{normalized_tier} builder encoded column schema hash mismatch"
        )

    expected_pre_model_hash = column_sequence_hash(builder.pre_model_columns_)
    if manifest.get("pre_model_column_schema_hash") != expected_pre_model_hash:
        raise RuntimeError(
            f"{normalized_tier} builder pre-model column schema hash mismatch"
        )

    expected_rare_hash = rare_map_schema_hash(builder.rare_category_maps_)
    if manifest.get("rare_map_schema_hash") != expected_rare_hash:
        raise RuntimeError(f"{normalized_tier} builder rare-map schema hash mismatch")

    processed_fingerprint = manifest.get("processed_manifest_fingerprint")
    if strict:
        if processed_manifest is None:
            raise RuntimeError("Strict builder validation requires a processed manifest")
        expected_fingerprint = processed_manifest.get("processed_manifest_fingerprint")
        if not expected_fingerprint:
            raise RuntimeError(
                "Processed manifest is missing processed_manifest_fingerprint"
            )
        if processed_fingerprint != expected_fingerprint:
            raise RuntimeError(
                f"{normalized_tier} builder lineage mismatch with processed manifest"
            )
    elif processed_manifest is not None:
        expected_fingerprint = processed_manifest.get("processed_manifest_fingerprint")
        if expected_fingerprint and processed_fingerprint and processed_fingerprint != expected_fingerprint:
            raise RuntimeError(
                f"{normalized_tier} builder lineage mismatch with processed manifest"
            )

    return {"warnings": warnings}


