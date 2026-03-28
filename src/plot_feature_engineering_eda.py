"""Reduced-tier EDA focused on application-only signal discovery.

This script is intentionally centered on the REDUCED model path so the EDA
matches the mentor-directed scope of the project. It highlights which
application-only features separate default risk well, how much missingness the
reduced tier carries, and how the reduced feature families are balanced.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_auc_score

if __package__ is None or __package__ == "":
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
else:
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

from src.feature_engineering import (
    APPLICATION_REQUIRED_INPUT_COLS,
    CATEGORICAL_MODEL_COLS,
    ENGINEERED_APP_FEATURE_COLS,
    REDUCED_FEATURE_FAMILY_COLS,
    _build_pre_model_frame,
)


DEFAULT_PROCESSED_DIR = "data/processed"
DEFAULT_OUTPUT_DIR = "notebooks/eda_plots/reduced_feature_engineering"


def _resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return Path(PROJECT_ROOT) / path


def _load_train_frame(processed_dir: str) -> pd.DataFrame:
    train_path = _resolve_repo_path(processed_dir) / "train.pkl"
    if not train_path.exists():
        raise FileNotFoundError(
            f"Missing processed training split: {train_path}. Run Module 1 first."
        )
    return pd.read_pickle(train_path)


def _ensure_output_dir(output_dir: str) -> Path:
    out = _resolve_repo_path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _savefig(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def _safe_univariate_auc(y_true: pd.Series, values: pd.Series) -> float:
    valid = y_true.notna() & values.notna()
    if valid.sum() < 100 or values[valid].nunique(dropna=True) < 2:
        return 0.5
    auc = float(roc_auc_score(y_true[valid], values[valid]))
    return max(auc, 1.0 - auc)


def _feature_family_for_column(column_name: str) -> str:
    for family_name, family_cols in REDUCED_FEATURE_FAMILY_COLS.items():
        if column_name in family_cols:
            return family_name
    return "OTHER"


def _build_numeric_signal_summary(reduced_pre_model_df: pd.DataFrame) -> pd.DataFrame:
    frame = reduced_pre_model_df.copy()
    numeric_cols = [
        c for c in frame.select_dtypes(include=[np.number]).columns if c != "TARGET"
    ]
    rows: list[dict[str, object]] = []
    for col in numeric_cols:
        target0 = frame.loc[frame["TARGET"] == 0, col]
        target1 = frame.loc[frame["TARGET"] == 1, col]
        rows.append(
            {
                "feature": col,
                "family": _feature_family_for_column(col),
                "missing_rate": float(frame[col].isna().mean()),
                "univariate_auc": _safe_univariate_auc(frame["TARGET"], frame[col]),
                "mean_no_default": float(target0.mean(skipna=True)),
                "mean_default": float(target1.mean(skipna=True)),
                "abs_target_gap": float(abs(target1.mean(skipna=True) - target0.mean(skipna=True))),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["univariate_auc", "abs_target_gap"],
        ascending=[False, False],
    )


def _plot_feature_family_counts(reduced_pre_model_df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for family_name, family_cols in REDUCED_FEATURE_FAMILY_COLS.items():
        present = [c for c in family_cols if c in reduced_pre_model_df.columns]
        rows.append(
            {
                "family": family_name.replace("_", " ").title(),
                "feature_count": len(present),
            }
        )
    summary = pd.DataFrame(rows).sort_values("feature_count", ascending=False)

    fig, ax = plt.subplots(figsize=(9, 4))
    sns.barplot(data=summary, x="family", y="feature_count", color="#2A9D8F", ax=ax)
    ax.set_title("Reduced Feature Families")
    ax.set_xlabel("")
    ax.set_ylabel("Feature count")
    ax.tick_params(axis="x", rotation=20)
    for i, row in summary.iterrows():
        ax.text(i, row["feature_count"], str(int(row["feature_count"])), ha="center", va="bottom")
    _savefig(output_dir / "reduced_feature_family_counts.png")
    return summary


def _plot_missingness(reduced_pre_model_df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    missing = (
        reduced_pre_model_df.drop(columns=["TARGET"], errors="ignore")
        .isna()
        .mean()
        .sort_values(ascending=False)
    )
    missing = (
        missing[missing > 0]
        .head(20)
        .rename_axis("feature")
        .reset_index(name="missing_rate")
    )
    if missing.empty:
        return missing

    fig, ax = plt.subplots(figsize=(12, 4))
    sns.barplot(data=missing, x="feature", y="missing_rate", color="#D85A30", ax=ax)
    ax.set_title("Reduced Feature Missingness (Top 20)")
    ax.set_xlabel("")
    ax.set_ylabel("Missing rate")
    ax.tick_params(axis="x", rotation=75, labelsize=8)
    _savefig(output_dir / "reduced_missingness_top20.png")
    return missing


def _plot_top_univariate_auc(signal_summary: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    top = signal_summary.head(15).sort_values("univariate_auc", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=top, x="univariate_auc", y="feature", color="#378ADD", ax=ax)
    ax.set_title("Top Reduced Features By Single-Feature ROC-AUC")
    ax.set_xlabel("Univariate ROC-AUC (directionless)")
    ax.set_ylabel("")
    _savefig(output_dir / "reduced_univariate_auc_top15.png")
    return top


def _plot_top_target_gap(signal_summary: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    top = signal_summary.sort_values("abs_target_gap", ascending=False).head(15).sort_values(
        "abs_target_gap",
        ascending=True,
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=top, x="abs_target_gap", y="feature", color="#D85A30", ax=ax)
    ax.set_title("Top Reduced Features By Target Separation")
    ax.set_xlabel("Absolute mean gap between default and non-default groups")
    ax.set_ylabel("")
    _savefig(output_dir / "reduced_target_gap_top15.png")
    return top


def _plot_correlation_heatmap(
    reduced_pre_model_df: pd.DataFrame,
    signal_summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    top_numeric = signal_summary.head(12)["feature"].tolist()
    corr_cols = [c for c in top_numeric if c in reduced_pre_model_df.columns]
    if len(corr_cols) < 3:
        return

    corr = reduced_pre_model_df[corr_cols].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation Heatmap For High-Signal Reduced Features")
    _savefig(output_dir / "reduced_feature_correlation_heatmap.png")


def _plot_top_engineered_distributions(
    reduced_pre_model_df: pd.DataFrame,
    signal_summary: pd.DataFrame,
    output_dir: Path,
) -> list[str]:
    engineered_top = [
        c
        for c in signal_summary["feature"].tolist()
        if c in ENGINEERED_APP_FEATURE_COLS and c in reduced_pre_model_df.columns
    ][:6]
    if not engineered_top:
        return []

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = np.atleast_1d(axes).reshape(2, 3)
    for ax, col in zip(axes.flat, engineered_top):
        sns.kdeplot(
            data=reduced_pre_model_df[[col, "TARGET"]].dropna(),
            x=col,
            hue="TARGET",
            common_norm=False,
            fill=True,
            ax=ax,
            palette=["#1D9E75", "#D85A30"],
        )
        ax.set_title(col)
        ax.set_xlabel("")
    for ax in axes.flat[len(engineered_top):]:
        ax.axis("off")
    fig.suptitle("High-Signal Engineered Features In The Reduced Tier", fontsize=14)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    plt.savefig(output_dir / "reduced_engineered_feature_distributions.png", dpi=160, bbox_inches="tight")
    plt.close()
    return engineered_top


def _plot_categorical_default_rates(train_df: pd.DataFrame, output_dir: Path) -> dict[str, list[dict[str, object]]]:
    summaries: dict[str, list[dict[str, object]]] = {}
    if "TARGET" not in train_df.columns:
        return summaries

    for col in CATEGORICAL_MODEL_COLS:
        if col not in train_df.columns:
            continue
        grouped = (
            train_df[[col, "TARGET"]]
            .copy()
            .assign(**{col: train_df[col].fillna("MISSING")})
            .groupby(col)["TARGET"]
            .agg(default_rate="mean", count="size")
            .sort_values(["default_rate", "count"], ascending=[False, False])
            .head(10)
            .reset_index()
        )
        summaries[col] = grouped.to_dict(orient="records")
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.barplot(data=grouped, x=col, y="default_rate", color="#7F77DD", ax=ax)
        ax.set_title(f"Default Rate By {col} (Reduced Tier Inputs)")
        ax.set_xlabel("")
        ax.set_ylabel("Default rate")
        ax.tick_params(axis="x", rotation=45)
        _savefig(output_dir / f"reduced_categorical_default_rate_{col.lower()}.png")
    return summaries


def generate_feature_engineering_eda(
    processed_dir: str,
    raw_dir: str,
    output_dir: str,
) -> None:
    del raw_dir  # Reduced-only EDA intentionally ignores FULL raw aggregates.
    sns.set_theme(style="whitegrid")
    output_path = _ensure_output_dir(output_dir)
    train_df = _load_train_frame(processed_dir)

    reduced_pre_model = _build_pre_model_frame(
        train_df,
        "REDUCED",
        raw_dir=None,
        allow_flattened_full_input=False,
    )
    reduced_pre_model["TARGET"] = train_df["TARGET"].to_numpy()

    signal_summary = _build_numeric_signal_summary(reduced_pre_model)
    family_summary = _plot_feature_family_counts(reduced_pre_model, output_path)
    missing_summary = _plot_missingness(reduced_pre_model, output_path)
    top_auc = _plot_top_univariate_auc(signal_summary, output_path)
    top_gap = _plot_top_target_gap(signal_summary, output_path)
    _plot_correlation_heatmap(reduced_pre_model, signal_summary, output_path)
    engineered_focus = _plot_top_engineered_distributions(reduced_pre_model, signal_summary, output_path)
    categorical_summaries = _plot_categorical_default_rates(train_df, output_path)

    summary = {
        "processed_dir": str(_resolve_repo_path(processed_dir).resolve()),
        "output_dir": str(output_path.resolve()),
        "input_application_columns": len(APPLICATION_REQUIRED_INPUT_COLS),
        "reduced_pre_model_feature_count": int(len([c for c in reduced_pre_model.columns if c != "TARGET"])),
        "engineered_feature_count": int(len([c for c in ENGINEERED_APP_FEATURE_COLS if c in reduced_pre_model.columns])),
        "engineered_features_present": [
            c for c in ENGINEERED_APP_FEATURE_COLS if c in reduced_pre_model.columns
        ],
        "feature_family_counts": family_summary.to_dict(orient="records"),
        "top_features_by_univariate_auc": top_auc.sort_values("univariate_auc", ascending=False).to_dict(orient="records"),
        "top_features_by_target_gap": top_gap.sort_values("abs_target_gap", ascending=False).to_dict(orient="records"),
        "top_missing_features": missing_summary.to_dict(orient="records"),
        "engineered_feature_focus": engineered_focus,
        "categorical_default_rate_highlights": categorical_summaries,
    }
    pd.Series(summary, dtype=object).to_json(output_path / "reduced_feature_engineering_eda_summary.json", indent=2)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate reduced-only EDA plots aligned to the application-only feature pipeline."
    )
    parser.add_argument(
        "--processed-dir",
        default=DEFAULT_PROCESSED_DIR,
        help="Directory containing processed splits from Module 1.",
    )
    parser.add_argument(
        "--raw-dir",
        default="data/raw",
        help="Accepted for backwards compatibility but ignored in reduced-only EDA mode.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write generated plots.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    generate_feature_engineering_eda(
        processed_dir=args.processed_dir,
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
    )
    print(f"Saved reduced feature-engineering EDA plots to {_resolve_repo_path(args.output_dir).resolve()}")
