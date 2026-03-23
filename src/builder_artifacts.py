"""Strict loading helpers for persisted FrozenFeatureBuilder artifacts."""

from __future__ import annotations

import os
from typing import Any, Mapping

import joblib

from src.feature_engineering import (
    FULL_FEATURE_BUILDER_ARTIFACT_PATH,
    REDUCED_FEATURE_BUILDER_ARTIFACT_PATH,
    FrozenFeatureBuilder,
)


def _resolve_builder_path(artifact_dir: str, tier: str) -> str:
    filename = os.path.basename(
        FULL_FEATURE_BUILDER_ARTIFACT_PATH
        if tier.upper() == "FULL"
        else REDUCED_FEATURE_BUILDER_ARTIFACT_PATH
    )
    return os.path.join(artifact_dir, filename)


def load_builder(
    *,
    tier: str,
    artifact_dir: str = "artifacts/",
    processed_dir: str | None = None,
    processed_manifest: Mapping[str, Any] | None = None,
    strict_artifacts: bool = True,
) -> FrozenFeatureBuilder:
    """Load one persisted builder and validate its basic contract."""

    _ = processed_dir, processed_manifest, strict_artifacts
    path = _resolve_builder_path(artifact_dir, tier)
    if not os.path.exists(path):
        raise RuntimeError(f"Missing required {tier.upper()} builder artifact: {path}")

    builder = joblib.load(path)
    if not isinstance(builder, FrozenFeatureBuilder):
        raise RuntimeError(f"{path} must contain a FrozenFeatureBuilder")
    if builder.tier.upper() != tier.upper():
        raise RuntimeError(
            f"{path} tier mismatch: expected {tier.upper()}, found {builder.tier}"
        )
    if not builder.encoded_columns_:
        raise RuntimeError(f"{path} has no frozen encoded columns")
    return builder


def load_validated_builders(
    *,
    artifact_dir: str = "artifacts/",
    processed_dir: str | None = None,
    processed_manifest: Mapping[str, Any] | None = None,
    strict_artifacts: bool = True,
) -> dict[str, FrozenFeatureBuilder]:
    """Load both builders for strict API startup."""

    return {
        "full_builder": load_builder(
            tier="FULL",
            artifact_dir=artifact_dir,
            processed_dir=processed_dir,
            processed_manifest=processed_manifest,
            strict_artifacts=strict_artifacts,
        ),
        "reduced_builder": load_builder(
            tier="REDUCED",
            artifact_dir=artifact_dir,
            processed_dir=processed_dir,
            processed_manifest=processed_manifest,
            strict_artifacts=strict_artifacts,
        ),
    }

