"""Canonical builder artifact loader and validator."""

from __future__ import annotations

import os
from typing import Any, Mapping

import joblib

from src.feature_engineering import FrozenFeatureBuilder
from src.runtime_verification import load_json_object, validate_builder_artifact

PROCESSED_MANIFEST_FILENAME = "processed_artifact_manifest.json"
_BUILDER_FILENAMES = {
    "FULL": ("full_feature_builder.joblib", "full_feature_builder.manifest.json"),
    "REDUCED": ("reduced_feature_builder.joblib", "reduced_feature_builder.manifest.json"),
}


def _normalize_tier(tier: str) -> str:
    normalized = str(tier).upper()
    if normalized not in _BUILDER_FILENAMES:
        raise ValueError(f"Unsupported builder tier: {tier!r}")
    return normalized


def _builder_paths(artifact_dir: str, tier: str) -> tuple[str, str]:
    joblib_name, manifest_name = _BUILDER_FILENAMES[_normalize_tier(tier)]
    return os.path.join(artifact_dir, joblib_name), os.path.join(artifact_dir, manifest_name)


def _resolve_processed_manifest(
    processed_dir: str,
    processed_manifest: Mapping[str, Any] | None,
    strict_artifacts: bool,
) -> Mapping[str, Any] | None:
    if processed_manifest is not None:
        return processed_manifest
    manifest_path = os.path.join(processed_dir, PROCESSED_MANIFEST_FILENAME)
    if not os.path.exists(manifest_path):
        if strict_artifacts:
            raise RuntimeError(f"Missing required processed manifest: {manifest_path}")
        return None
    return load_json_object(manifest_path, "processed manifest")


def load_builder(
    *,
    tier: str,
    artifact_dir: str,
    processed_dir: str,
    processed_manifest: Mapping[str, Any] | None = None,
    strict_artifacts: bool = True,
) -> FrozenFeatureBuilder:
    normalized_tier = _normalize_tier(tier)
    builder_path, manifest_path = _builder_paths(artifact_dir, normalized_tier)
    if not os.path.exists(builder_path):
        raise RuntimeError(f"Missing required {normalized_tier} builder artifact: {builder_path}")

    try:
        builder = joblib.load(builder_path)
    except Exception as exc:
        raise RuntimeError(f"Failed to load {normalized_tier} builder artifact: {builder_path}") from exc

    manifest: Mapping[str, Any] | None = None
    if os.path.exists(manifest_path):
        manifest = load_json_object(manifest_path, f"{normalized_tier} builder manifest")
    elif strict_artifacts:
        raise RuntimeError(f"Missing required {normalized_tier} builder manifest: {manifest_path}")

    resolved_processed_manifest = _resolve_processed_manifest(
        processed_dir,
        processed_manifest,
        strict_artifacts,
    )
    validate_builder_artifact(
        builder,
        tier=normalized_tier,
        manifest=manifest,
        processed_manifest=resolved_processed_manifest,
        strict=strict_artifacts,
    )
    return builder


def load_full_builder(
    *,
    artifact_dir: str,
    processed_dir: str,
    processed_manifest: Mapping[str, Any] | None = None,
    strict_artifacts: bool = True,
) -> FrozenFeatureBuilder:
    return load_builder(
        tier="FULL",
        artifact_dir=artifact_dir,
        processed_dir=processed_dir,
        processed_manifest=processed_manifest,
        strict_artifacts=strict_artifacts,
    )


def load_reduced_builder(
    *,
    artifact_dir: str,
    processed_dir: str,
    processed_manifest: Mapping[str, Any] | None = None,
    strict_artifacts: bool = True,
) -> FrozenFeatureBuilder:
    return load_builder(
        tier="REDUCED",
        artifact_dir=artifact_dir,
        processed_dir=processed_dir,
        processed_manifest=processed_manifest,
        strict_artifacts=strict_artifacts,
    )


def load_validated_builders(
    *,
    artifact_dir: str,
    processed_dir: str,
    processed_manifest: Mapping[str, Any] | None = None,
    strict_artifacts: bool = True,
) -> dict[str, FrozenFeatureBuilder]:
    resolved_processed_manifest = _resolve_processed_manifest(
        processed_dir,
        processed_manifest,
        strict_artifacts,
    )
    return {
        "FULL": load_builder(
            tier="FULL",
            artifact_dir=artifact_dir,
            processed_dir=processed_dir,
            processed_manifest=resolved_processed_manifest,
            strict_artifacts=strict_artifacts,
        ),
        "REDUCED": load_builder(
            tier="REDUCED",
            artifact_dir=artifact_dir,
            processed_dir=processed_dir,
            processed_manifest=resolved_processed_manifest,
            strict_artifacts=strict_artifacts,
        ),
    }
