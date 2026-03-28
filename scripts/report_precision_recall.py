from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score

from configs.config import ARTIFACT_DIR, DATA_DIR, DECLINE_THRESHOLD
from src.builder_artifacts import load_validated_builders
from src.feature_engineering import DEFAULT_FULL_FEATURE_VIEW, build_full, build_reduced


TIER_PREFIX = {
    "FULL": "full",
    "REDUCED": "reduced",
}


def _resolve_repo_path(path_str: str) -> Path:
    candidate = Path(path_str)
    if candidate.is_absolute():
        return candidate.resolve()
    return (REPO_ROOT / candidate).resolve()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute precision and recall for a persisted tier model on a processed split."
    )
    parser.add_argument(
        "--tier",
        choices=("FULL", "REDUCED"),
        default="FULL",
        help="Tier to evaluate. Defaults to FULL.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "val_model", "val_policy", "test"),
        default="test",
        help="Processed split to score. Defaults to test.",
    )
    parser.add_argument(
        "--artifact-dir",
        default=ARTIFACT_DIR,
        help=f"Artifact directory. Defaults to {ARTIFACT_DIR!r}.",
    )
    parser.add_argument(
        "--processed-dir",
        default=DATA_DIR,
        help=f"Processed data directory. Defaults to {DATA_DIR!r}.",
    )
    parser.add_argument(
        "--raw-dir",
        default=None,
        help="Raw data directory for FULL feature construction. Defaults to the report value or data/raw.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DECLINE_THRESHOLD,
        help=f"Binary classification threshold applied to calibrated PD. Defaults to {DECLINE_THRESHOLD}.",
    )
    return parser.parse_args()


def _required_artifacts(tier: str) -> list[str]:
    prefix = TIER_PREFIX[tier]
    return [
        f"{prefix}_model.joblib",
        f"{prefix}_calibrator.joblib",
        f"{prefix}_feature_builder.joblib",
    ]


def _predict_raw_pd(model: Any, features: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_raw_pd") and callable(model.predict_raw_pd):
        raw = model.predict_raw_pd(features)
    elif hasattr(model, "predict_proba") and callable(model.predict_proba):
        raw = np.asarray(model.predict_proba(features), dtype=float)[:, 1]
    else:
        raise RuntimeError("Loaded model exposes neither predict_raw_pd nor predict_proba.")
    values = np.asarray(raw, dtype=float).reshape(-1)
    if values.shape[0] != len(features):
        raise RuntimeError("Model output row count does not match feature rows.")
    return values


def _predict_calibrated_pd(calibrator: Any, raw_pd: np.ndarray) -> np.ndarray:
    if not hasattr(calibrator, "predict") or not callable(calibrator.predict):
        raise RuntimeError("Loaded calibrator does not expose predict.")
    values = np.asarray(calibrator.predict(np.asarray(raw_pd, dtype=float)), dtype=float).reshape(-1)
    if values.shape[0] != len(raw_pd):
        raise RuntimeError("Calibrator output row count does not match raw probabilities.")
    return np.clip(values, 0.0, 1.0)


def _resolve_raw_dir(args: argparse.Namespace, report: dict[str, Any]) -> Path:
    if args.raw_dir:
        return _resolve_repo_path(args.raw_dir)
    report_raw = report.get("raw_dir")
    if isinstance(report_raw, str) and report_raw:
        return _resolve_repo_path(report_raw)
    return _resolve_repo_path("data/raw")


def _load_features(
    *,
    tier: str,
    split_df: pd.DataFrame,
    artifact_dir: Path,
    processed_dir: Path,
    raw_dir: Path,
    report: dict[str, Any],
) -> pd.DataFrame:
    builders = load_validated_builders(
        artifact_dir=str(artifact_dir),
        processed_dir=str(processed_dir),
        strict_artifacts=False,
    )
    builder = builders[tier]
    if tier == "FULL":
        tier_report = report.get("tiers", {}).get("FULL", {}) if isinstance(report.get("tiers"), dict) else {}
        feature_view = tier_report.get("feature_view", DEFAULT_FULL_FEATURE_VIEW)
        raw_dir_arg = str(raw_dir) if raw_dir.exists() else None
        return build_full(
            split_df,
            builder,
            raw_dir=raw_dir_arg,
            feature_view=str(feature_view),
        )
    return build_reduced(split_df, builder)


def main() -> int:
    args = _parse_args()
    tier = args.tier.upper()
    artifact_dir = _resolve_repo_path(args.artifact_dir)
    processed_dir = _resolve_repo_path(args.processed_dir)
    try:
        report = _load_json(artifact_dir / "reproducibility_report.json")
        raw_dir = _resolve_raw_dir(args, report)

        missing_artifacts = [
            str(artifact_dir / name)
            for name in _required_artifacts(tier)
            if not (artifact_dir / name).exists()
        ]
        if missing_artifacts:
            print(
                json.dumps(
                    {
                        "status": "missing_artifacts",
                        "tier": tier,
                        "missing": missing_artifacts,
                    },
                    indent=2,
                )
            )
            return 1

        split_path = processed_dir / f"{args.split}.pkl"
        if not split_path.exists():
            raise FileNotFoundError(f"Missing processed split: {split_path}")

        split_df = pd.read_pickle(split_path)
        if "TARGET" not in split_df.columns:
            raise RuntimeError(f"{split_path} does not contain a TARGET column.")

        features = _load_features(
            tier=tier,
            split_df=split_df,
            artifact_dir=artifact_dir,
            processed_dir=processed_dir,
            raw_dir=raw_dir,
            report=report,
        )

        prefix = TIER_PREFIX[tier]
        model = joblib.load(artifact_dir / f"{prefix}_model.joblib")
        calibrator = joblib.load(artifact_dir / f"{prefix}_calibrator.joblib")

        y_true = split_df["TARGET"].to_numpy(dtype=int)
        raw_pd = _predict_raw_pd(model, features)
        calibrated_pd = _predict_calibrated_pd(calibrator, raw_pd)
        y_pred = (calibrated_pd >= float(args.threshold)).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        tier_report = report.get("tiers", {}).get(tier.lower()) if isinstance(report.get("tiers"), dict) else None
        model_version = None
        if isinstance(tier_report, dict):
            model_version = tier_report.get("model_version")
        if not model_version:
            model_version = report.get(f"{prefix}_model_version")

        payload = {
            "status": "ok",
            "tier": tier,
            "split": args.split,
            "threshold": float(args.threshold),
            "model_version": model_version,
            "artifact_dir": str(artifact_dir),
            "processed_dir": str(processed_dir),
            "raw_dir": str(raw_dir),
            "rows": int(len(y_true)),
            "positive_labels": int(y_true.sum()),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "confusion_matrix": {
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            },
        }
        print(json.dumps(payload, indent=2))
        return 0
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "tier": tier,
                    "split": args.split,
                    "message": str(exc),
                },
                indent=2,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
