from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import joblib
import pandas as pd

DEFAULT_FULL_BUILDER_PATH = "artifacts/full_feature_builder.joblib"
DEFAULT_REDUCED_BUILDER_PATH = "artifacts/reduced_feature_builder.joblib"
DEFAULT_PROCESSED_MANIFEST_PATH = os.path.join(
    "data",
    "processed",
    "processed_artifact_manifest.json",
)
MANIFEST_SUFFIX = ".manifest.json"
QUARANTINE_DIR_NAME = "quarantine"
RECOMMENDED_MIN_ENCODED_COLUMNS = {"FULL": 150, "REDUCED": 100}


class BuilderValidationError(ValueError):
    """Raised when a persisted builder artifact fails strict validation."""


def builder_manifest_path(builder_path: str) -> str:
    base, ext = os.path.splitext(builder_path)
    if ext.lower() == ".joblib":
        return f"{base}{MANIFEST_SUFFIX}"
    return f"{builder_path}{MANIFEST_SUFFIX}"


def default_builder_path_for_tier(tier: str) -> str:
    tier_upper = tier.upper()
    if tier_upper == "FULL":
        return DEFAULT_FULL_BUILDER_PATH
    if tier_upper == "REDUCED":
        return DEFAULT_REDUCED_BUILDER_PATH
    raise ValueError(f"Unsupported tier: {tier}")


def current_git_commit(repo_root: str = ".") -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def hash_ordered_values(values: list[str]) -> str:
    payload = "\n".join(values).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def canonical_json_hash(payload: Any) -> str:
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def dataframe_schema_hash(df: pd.DataFrame) -> str:
    tokens = [f"{column}:{df.dtypes[column]}" for column in df.columns]
    return hash_ordered_values(tokens)


def dataset_fingerprint(
    fit_df: pd.DataFrame,
    id_col: str = "SK_ID_CURR",
    target_col: str = "TARGET",
) -> str:
    missing = [c for c in [id_col, target_col] if c not in fit_df.columns]
    if missing:
        raise ValueError(f"Missing columns for dataset fingerprint: {sorted(missing)}")
    ordered = fit_df[[id_col, target_col]].sort_values(id_col, kind="mergesort")
    payload = ordered.to_csv(
        index=False,
        header=False,
        lineterminator="\n",
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def rare_map_sizes(rare_maps: dict[str, set[str]]) -> dict[str, int]:
    return {
        key: int(len(values))
        for key, values in sorted(rare_maps.items(), key=lambda kv: kv[0])
    }


def rare_map_schema_hash(rare_maps: dict[str, set[str]]) -> str:
    ordered_tokens: list[str] = []
    for key in sorted(rare_maps):
        ordered_tokens.append(f"{key}::" + "||".join(sorted(map(str, rare_maps[key]))))
    return hash_ordered_values(ordered_tokens)


def processed_manifest_path(data_processed_dir: str = "data/processed") -> str:
    return os.path.join(data_processed_dir, os.path.basename(DEFAULT_PROCESSED_MANIFEST_PATH))


def processed_manifest_fingerprint(manifest: dict[str, Any]) -> str:
    payload = {
        key: value
        for key, value in manifest.items()
        if key != "processed_manifest_fingerprint"
    }
    return canonical_json_hash(payload)


def validate_processed_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    expected = processed_manifest_fingerprint(manifest)
    actual = manifest.get("processed_manifest_fingerprint")
    if actual != expected:
        raise BuilderValidationError(
            "Processed manifest fingerprint mismatch: "
            f"expected {expected!r}, found {actual!r}"
        )
    return manifest


def load_processed_artifact_manifest(
    manifest_path: str = DEFAULT_PROCESSED_MANIFEST_PATH,
) -> dict[str, Any]:
    if not os.path.exists(manifest_path):
        raise BuilderValidationError(f"Missing processed manifest: {manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    return validate_processed_manifest(manifest)


def quarantine_existing_paths(
    paths: list[str],
    quarantine_root: str,
) -> dict[str, str]:
    existing_paths = [path for path in paths if os.path.exists(path)]
    if not existing_paths:
        return {}

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    batch_dir = os.path.join(quarantine_root, timestamp)
    os.makedirs(batch_dir, exist_ok=True)

    moved: dict[str, str] = {}
    for path in existing_paths:
        base_name = os.path.basename(path.rstrip(os.sep))
        target = os.path.join(batch_dir, base_name)
        suffix = 1
        while os.path.exists(target):
            target = os.path.join(batch_dir, f"{base_name}_{suffix}")
            suffix += 1
        shutil.move(path, target)
        moved[path] = target

    return moved


def create_builder_manifest(
    builder: Any,
    *,
    fit_df: pd.DataFrame,
    fit_split_name: str,
    feature_engineering_version: str,
    aggregate_contract_version: str,
    git_commit: str | None = None,
    processed_manifest_fingerprint: str | None = None,
    id_col: str = "SK_ID_CURR",
    target_col: str = "TARGET",
) -> dict[str, Any]:
    tier = str(getattr(builder, "tier", "")).upper()
    encoded_columns = list(getattr(builder, "encoded_columns_", []))
    pre_model_columns = list(getattr(builder, "pre_model_columns_", []))
    categorical_columns = list(getattr(builder, "categorical_columns_", []))
    rare_maps = getattr(builder, "rare_category_maps_", {})

    return {
        "manifest_version": 2,
        "builder_tier": tier,
        "fit_split_name": fit_split_name,
        "fit_row_count": int(len(fit_df)),
        "dataset_fingerprint": dataset_fingerprint(fit_df, id_col=id_col, target_col=target_col),
        "feature_engineering_version": feature_engineering_version,
        "git_commit": git_commit or current_git_commit(),
        "processed_manifest_fingerprint": processed_manifest_fingerprint,
        "aggregate_contract_version": aggregate_contract_version,
        "encoded_column_schema_hash": hash_ordered_values(encoded_columns),
        "pre_model_column_schema_hash": hash_ordered_values(pre_model_columns),
        "fit_timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "encoded_column_count": int(len(encoded_columns)),
        "rare_map_sizes": rare_map_sizes(rare_maps),
        "categorical_columns": categorical_columns,
        "rare_map_schema_hash": rare_map_schema_hash(rare_maps),
    }


def _ensure_not_quarantine_path(builder_path: str) -> None:
    path_parts = {part.lower() for part in Path(builder_path).parts}
    if QUARANTINE_DIR_NAME.lower() in path_parts:
        raise BuilderValidationError(
            f"Refusing to load or publish builder artifact inside quarantine: {builder_path}"
        )


def _assert_manifest_value(manifest: dict[str, Any], key: str, expected: Any) -> None:
    actual = manifest.get(key)
    if actual != expected:
        raise BuilderValidationError(
            f"Builder manifest mismatch for {key}: expected {expected!r}, found {actual!r}"
        )


def _assert_builder_has_required_attributes(builder: Any) -> None:
    required_attrs = [
        "tier",
        "encoded_columns_",
        "pre_model_columns_",
        "rare_category_maps_",
        "aggregate_feature_cols_",
        "categorical_columns_",
    ]
    missing = [attr for attr in required_attrs if not hasattr(builder, attr)]
    if missing:
        raise BuilderValidationError(
            f"Loaded builder artifact is missing required attributes: {sorted(missing)}"
        )


def load_builder_artifact(
    builder_path: str,
    *,
    fit_df: pd.DataFrame,
    expected_tier: str,
    fit_split_name: str,
    feature_engineering_version: str,
    aggregate_contract_version: str,
    expected_aggregate_feature_count: int,
    require_git_match: bool = False,
    expected_rare_map_sizes: dict[str, int] | None = None,
    expected_processed_manifest_fingerprint: str | None = None,
) -> tuple[Any, dict[str, Any], list[str]]:
    _ensure_not_quarantine_path(builder_path)
    manifest_path = builder_manifest_path(builder_path)
    if not os.path.exists(builder_path):
        raise BuilderValidationError(f"Missing builder artifact: {builder_path}")
    if not os.path.exists(manifest_path):
        raise BuilderValidationError(
            f"Missing builder manifest for {builder_path}. Rebuild from train.pkl."
        )

    builder = joblib.load(builder_path)
    _assert_builder_has_required_attributes(builder)

    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    tier_upper = expected_tier.upper()
    _assert_manifest_value(manifest, "builder_tier", tier_upper)
    _assert_manifest_value(manifest, "fit_split_name", fit_split_name)
    _assert_manifest_value(manifest, "fit_row_count", int(len(fit_df)))
    _assert_manifest_value(manifest, "dataset_fingerprint", dataset_fingerprint(fit_df))
    _assert_manifest_value(manifest, "feature_engineering_version", feature_engineering_version)
    _assert_manifest_value(
        manifest,
        "processed_manifest_fingerprint",
        expected_processed_manifest_fingerprint,
    )
    _assert_manifest_value(manifest, "aggregate_contract_version", aggregate_contract_version)
    _assert_manifest_value(
        manifest,
        "encoded_column_schema_hash",
        hash_ordered_values(list(builder.encoded_columns_)),
    )
    _assert_manifest_value(
        manifest,
        "pre_model_column_schema_hash",
        hash_ordered_values(list(builder.pre_model_columns_)),
    )
    _assert_manifest_value(
        manifest,
        "rare_map_schema_hash",
        rare_map_schema_hash(getattr(builder, "rare_category_maps_", {})),
    )

    if require_git_match:
        _assert_manifest_value(manifest, "git_commit", current_git_commit())

    aggregate_feature_count = int(len(getattr(builder, "aggregate_feature_cols_", [])))
    if aggregate_feature_count != expected_aggregate_feature_count:
        raise BuilderValidationError(
            "Builder aggregate contract mismatch: "
            f"expected {expected_aggregate_feature_count}, found {aggregate_feature_count}"
        )

    rare_sizes = rare_map_sizes(getattr(builder, "rare_category_maps_", {}))
    if rare_sizes and all(size == 0 for size in rare_sizes.values()):
        raise BuilderValidationError(
            "Builder artifact collapsed all application categoricals; rebuild from real train data."
        )

    warnings_out: list[str] = []
    min_encoded_cols = RECOMMENDED_MIN_ENCODED_COLUMNS.get(tier_upper)
    if min_encoded_cols is not None and len(builder.encoded_columns_) < min_encoded_cols:
        warnings_out.append(
            f"{tier_upper} encoded column count {len(builder.encoded_columns_)} "
            f"is below the recommended threshold {min_encoded_cols}"
        )

    if expected_rare_map_sizes is not None and rare_sizes != expected_rare_map_sizes:
        warnings_out.append(
            "Rare-map sizes differ from the expected reference profile: "
            f"expected {expected_rare_map_sizes}, found {rare_sizes}"
        )

    for warning_message in warnings_out:
        warnings.warn(warning_message, stacklevel=2)

    return builder, manifest, warnings_out


def save_builder_artifact(
    builder: Any,
    builder_path: str,
    *,
    fit_df: pd.DataFrame,
    fit_split_name: str,
    feature_engineering_version: str,
    aggregate_contract_version: str,
    expected_aggregate_feature_count: int,
    strict_validation: bool = False,
    processed_manifest_fingerprint: str | None = None,
) -> dict[str, Any]:
    _ensure_not_quarantine_path(builder_path)
    _assert_builder_has_required_attributes(builder)

    builder_dir = os.path.dirname(builder_path) or "."
    os.makedirs(builder_dir, exist_ok=True)
    manifest_path = builder_manifest_path(builder_path)

    manifest = create_builder_manifest(
        builder,
        fit_df=fit_df,
        fit_split_name=fit_split_name,
        feature_engineering_version=feature_engineering_version,
        aggregate_contract_version=aggregate_contract_version,
        processed_manifest_fingerprint=processed_manifest_fingerprint,
    )

    builder_fd, tmp_builder_path = tempfile.mkstemp(
        dir=builder_dir,
        prefix=".builder_tmp_",
        suffix=".joblib",
    )
    os.close(builder_fd)
    tmp_manifest_path = builder_manifest_path(tmp_builder_path)

    try:
        joblib.dump(builder, tmp_builder_path)
        with open(tmp_manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)

        validate_builder_pair(
            tmp_builder_path,
            tmp_manifest_path,
            fit_df=fit_df,
            expected_tier=str(getattr(builder, "tier", "")).upper(),
            fit_split_name=fit_split_name,
            feature_engineering_version=feature_engineering_version,
            aggregate_contract_version=aggregate_contract_version,
            expected_aggregate_feature_count=expected_aggregate_feature_count,
            strict_validation=strict_validation,
            processed_manifest_fingerprint=processed_manifest_fingerprint,
        )

        os.replace(tmp_builder_path, builder_path)
        os.replace(tmp_manifest_path, manifest_path)
    finally:
        if os.path.exists(tmp_builder_path):
            os.remove(tmp_builder_path)
        if os.path.exists(tmp_manifest_path):
            os.remove(tmp_manifest_path)

    return manifest


def validate_builder_pair(
    builder_path: str,
    manifest_path: str,
    *,
    fit_df: pd.DataFrame,
    expected_tier: str,
    fit_split_name: str,
    feature_engineering_version: str,
    aggregate_contract_version: str,
    expected_aggregate_feature_count: int,
    strict_validation: bool = False,
    processed_manifest_fingerprint: str | None = None,
) -> tuple[Any, dict[str, Any], list[str]]:
    if not os.path.exists(manifest_path):
        raise BuilderValidationError(f"Missing manifest during validation: {manifest_path}")

    if strict_validation:
        return load_builder_artifact(
            builder_path,
            fit_df=fit_df,
            expected_tier=expected_tier,
            fit_split_name=fit_split_name,
            feature_engineering_version=feature_engineering_version,
            aggregate_contract_version=aggregate_contract_version,
            expected_aggregate_feature_count=expected_aggregate_feature_count,
            expected_processed_manifest_fingerprint=processed_manifest_fingerprint,
        )

    builder = joblib.load(builder_path)
    _assert_builder_has_required_attributes(builder)
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    _assert_manifest_value(manifest, "builder_tier", expected_tier.upper())
    _assert_manifest_value(manifest, "fit_split_name", fit_split_name)
    _assert_manifest_value(manifest, "fit_row_count", int(len(fit_df)))
    _assert_manifest_value(manifest, "dataset_fingerprint", dataset_fingerprint(fit_df))
    _assert_manifest_value(manifest, "feature_engineering_version", feature_engineering_version)
    _assert_manifest_value(
        manifest,
        "processed_manifest_fingerprint",
        processed_manifest_fingerprint,
    )
    _assert_manifest_value(manifest, "aggregate_contract_version", aggregate_contract_version)
    _assert_manifest_value(
        manifest,
        "encoded_column_schema_hash",
        hash_ordered_values(list(builder.encoded_columns_)),
    )
    _assert_manifest_value(
        manifest,
        "pre_model_column_schema_hash",
        hash_ordered_values(list(builder.pre_model_columns_)),
    )
    return builder, manifest, []


def _artifact_path(artifact_dir: str, default_path: str) -> str:
    return os.path.join(artifact_dir, os.path.basename(default_path))


def _load_train_split(processed_dir: str) -> pd.DataFrame:
    train_path = os.path.join(processed_dir, "train.pkl")
    if not os.path.exists(train_path):
        raise RuntimeError(f"Missing required processed split: {train_path}")
    train_df = pd.read_pickle(train_path)
    if not isinstance(train_df, pd.DataFrame):
        raise RuntimeError(f"Processed split must be a pandas DataFrame: {train_path}")
    return train_df


def _load_builder_for_tier(
    tier: str,
    *,
    artifact_dir: str,
    processed_dir: str,
    processed_manifest: Mapping[str, Any] | None,
    strict_artifacts: bool,
    require_git_match: bool = False,
) -> Any:
    from src.feature_engineering import (
        AGGREGATE_CONTRACT_VERSION,
        ALL_AGGREGATE_FEATURE_COLS,
        FEATURE_ENGINEERING_VERSION,
    )

    train_df = _load_train_split(processed_dir)
    builder_path = _artifact_path(artifact_dir, default_builder_path_for_tier(tier))
    expected_processed_manifest_fingerprint = None
    if processed_manifest is not None:
        candidate = processed_manifest.get("processed_manifest_fingerprint")
        if candidate is not None:
            expected_processed_manifest_fingerprint = str(candidate)

    try:
        if strict_artifacts:
            builder, _, _ = load_builder_artifact(
                builder_path,
                fit_df=train_df,
                expected_tier=tier,
                fit_split_name="train",
                feature_engineering_version=FEATURE_ENGINEERING_VERSION,
                aggregate_contract_version=AGGREGATE_CONTRACT_VERSION,
                expected_aggregate_feature_count=(
                    len(ALL_AGGREGATE_FEATURE_COLS) if tier.upper() == "FULL" else 0
                ),
                require_git_match=require_git_match,
                expected_processed_manifest_fingerprint=expected_processed_manifest_fingerprint,
            )
            return builder
        builder = joblib.load(builder_path)
        _assert_builder_has_required_attributes(builder)
        return builder
    except Exception as exc:
        raise RuntimeError(f"Failed to load validated {tier.upper()} builder: {builder_path}") from exc


def load_validated_builders(
    *,
    artifact_dir: str = "artifacts/",
    processed_dir: str = "data/processed/",
    processed_manifest: Mapping[str, Any] | None = None,
    strict_artifacts: bool = True,
    require_git_match: bool = False,
    **_: Any,
) -> dict[str, Any]:
    return {
        "full_builder": _load_builder_for_tier(
            "FULL",
            artifact_dir=artifact_dir,
            processed_dir=processed_dir,
            processed_manifest=processed_manifest,
            strict_artifacts=strict_artifacts,
            require_git_match=require_git_match,
        ),
        "reduced_builder": _load_builder_for_tier(
            "REDUCED",
            artifact_dir=artifact_dir,
            processed_dir=processed_dir,
            processed_manifest=processed_manifest,
            strict_artifacts=strict_artifacts,
            require_git_match=require_git_match,
        ),
    }


def load_builder_runtime(**kwargs: Any) -> dict[str, Any]:
    return load_validated_builders(**kwargs)


def load_builders(**kwargs: Any) -> dict[str, Any]:
    return load_validated_builders(**kwargs)


def load_builder(
    tier: str,
    *,
    artifact_dir: str = "artifacts/",
    processed_dir: str = "data/processed/",
    processed_manifest: Mapping[str, Any] | None = None,
    strict_artifacts: bool = True,
    require_git_match: bool = False,
    **_: Any,
) -> Any:
    return _load_builder_for_tier(
        tier,
        artifact_dir=artifact_dir,
        processed_dir=processed_dir,
        processed_manifest=processed_manifest,
        strict_artifacts=strict_artifacts,
        require_git_match=require_git_match,
    )
