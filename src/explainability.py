"""Module 4 — Explainability.

SHAP global + local plots, business-language reason mapping,
evaluation plots (confusion matrix, ROC curve, F-beta sweep).
"""

from __future__ import annotations

import os
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────

REASON_MAP: dict[str, str] = {
    "CREDIT_INCOME_RATIO": "Requested credit is high relative to stated income",
    "ANNUITY_INCOME_RATIO": "Repayment burden is high relative to stated income",
    "INST_DPD": "Past installment payments show late-payment behavior",
    "BUREAU_": "External credit history indicates elevated repayment risk",
    "CC_": "Credit card utilization or delinquency history indicates elevated risk",
    "POS_": "Past point-of-sale loan behavior indicates elevated risk",
    "mock_ratio": "high debt-to-income ratio",
    "mock_credit": "large requested loan amount",
    "mock_income": "income level is not strong relative to the borrowing request",
    "mock_ext_mean": "external credit profile looks weaker than ideal",
    "mock_social": "social-circle indicators suggest higher repayment risk",
    "mock_bureau": "bureau history suggests elevated repayment risk",
    "mock_previous": "previous application history suggests elevated repayment risk",
    "mock_installments": "installment repayment history raises concern",
    "mock_pos": "point-of-sale borrowing history raises concern",
    "mock_cc": "credit card borrowing behavior raises concern",
}

DEFAULT_REASON: str = (
    "Combined application and repayment profile increased model risk"
)

SHAP_PLOTS_DIR = os.environ.get("SHAP_PLOTS_DIR", "notebooks/shap_plots/")
EVAL_PLOTS_DIR = os.environ.get("EVAL_PLOTS_DIR", "notebooks/eval_plots/")


# ──────────────────────────────────────────────────────────────────────
# Stage 1 — Reason Mapping & top_5_explanations
# ──────────────────────────────────────────────────────────────────────

def render_reason(feature_name: str) -> str:
    """Prefix-match *feature_name* against REASON_MAP keys (case-sensitive)."""
    for key, reason in REASON_MAP.items():
        if feature_name.startswith(key):
            return reason
    return DEFAULT_REASON


def top_5_explanations_from_shap(
    shap_series: pd.Series,
) -> list[dict[str, str]]:
    """Return top-5 features by |SHAP| with business reasons."""
    if len(shap_series) == 0:
        raise ValueError("shap_series is empty — cannot extract top 5")
    top = shap_series.abs().sort_values(ascending=False).head(5)
    return [
        {"feature": feat, "reason": render_reason(feat)}
        for feat in top.index
    ]


# ──────────────────────────────────────────────────────────────────────
# Stage 2 — SHAP Plot Generation
# ──────────────────────────────────────────────────────────────────────

def generate_shap_plots(
    explainer: Any,
    X_test: np.ndarray,
    feature_names: list[str],
    calibrated_pds: np.ndarray,
    output_dir: str = SHAP_PLOTS_DIR,
) -> None:
    """Generate and save 6 SHAP PNG plots."""
    os.makedirs(output_dir, exist_ok=True)
    import shap

    # Step A — Compute SHAP values
    shap_values = explainer(X_test)
    is_explanation = isinstance(shap_values, shap.Explanation)

    if is_explanation:
        sv_array = shap_values.values
    else:
        sv_array = np.array(shap_values)
        if sv_array.ndim == 3:
            sv_array = sv_array[:, :, 1]

    sv_df = pd.DataFrame(sv_array, columns=feature_names)

    # Step B — Plot 1: global_summary_bar.png
    mean_abs_shap = sv_df.abs().mean().sort_values(ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(
        mean_abs_shap.index[::-1],
        mean_abs_shap.values[::-1],
        color="#7F77DD",
    )
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Global feature importance — top 20")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "global_summary_bar.png"), dpi=150)
    plt.close()

    # Step C — Plot 2: beeswarm.png
    try:
        top20_names = sv_df.abs().mean().nlargest(20).index.tolist()
        top20_idx = [feature_names.index(n) for n in top20_names if n in feature_names]
        shap_exp = shap.Explanation(
            values=sv_array[:, top20_idx],
            data=X_test[:, top20_idx],
            feature_names=[feature_names[i] for i in top20_idx],
        )
        fig = plt.figure(figsize=(10, 7))
        shap.plots.beeswarm(shap_exp, show=False, max_display=20)
        plt.title("SHAP beeswarm — top 20 features")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "beeswarm.png"), dpi=150)
        plt.close()
    except Exception:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(
            0.5, 0.5,
            "Beeswarm not available for this explainer type",
            ha="center", va="center", fontsize=12,
        )
        ax.axis("off")
        plt.savefig(os.path.join(output_dir, "beeswarm.png"), dpi=100)
        plt.close()

    # Step D — Plot 3: heatmap.png
    sample = sv_df.iloc[: min(500, len(sv_df))]
    top20_cols = sv_df.abs().mean().nlargest(20).index.tolist()
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(
        sample[top20_cols].T,
        cmap="coolwarm",
        center=0,
        ax=ax,
        xticklabels=False,
        cbar_kws={"label": "SHAP value"},
    )
    ax.set_title("SHAP value heatmap — top 20 features (sample of test set)")
    ax.set_ylabel("Feature")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "heatmap.png"), dpi=150)
    plt.close()

    # Step E — Local force plots (3 plots)
    cases = {
        "approve": calibrated_pds < 0.15,
        "review": (calibrated_pds >= 0.15) & (calibrated_pds < 0.35),
        "decline": calibrated_pds >= 0.35,
    }
    for case, mask in cases.items():
        matching = np.where(mask)[0]
        if len(matching) > 0:
            row_idx = matching[0]
        else:
            print(f"WARNING: no {case} example found in test set, using index 0")
            row_idx = 0

        shap_row = pd.Series(sv_array[row_idx], index=feature_names)
        top10 = shap_row.abs().nlargest(10)
        top10_vals = shap_row[top10.index]
        colors = ["#D85A30" if v > 0 else "#1D9E75" for v in top10_vals.values]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(
            top10.index[::-1],
            top10_vals.values[::-1],
            color=colors[::-1],
        )
        ax.axvline(x=0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("SHAP value")
        ax.set_title(
            f"Local explanation — {case.upper()} "
            f"(PD={calibrated_pds[row_idx]:.3f})"
        )
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"local_{case}.png"), dpi=150)
        plt.close()

    print(f"All 6 SHAP plots saved to {output_dir}")


# ──────────────────────────────────────────────────────────────────────
# Stage 3 — Eval Plots
# ──────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    y_true: np.ndarray,
    calibrated_pds: np.ndarray,
    threshold: float = 0.35,
    output_dir: str = EVAL_PLOTS_DIR,
) -> None:
    """Plot and save a confusion matrix PNG."""
    os.makedirs(output_dir, exist_ok=True)
    from sklearn.metrics import confusion_matrix

    y_pred = (calibrated_pds >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        xticklabels=["Predicted 0", "Predicted 1"],
        yticklabels=["Actual 0", "Actual 1"],
    )
    ax.set_title(f"Confusion matrix (threshold={threshold:.2f})")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=150)
    plt.close()


def plot_roc_curve(
    y_true: np.ndarray,
    calibrated_pds: np.ndarray,
    output_dir: str = EVAL_PLOTS_DIR,
) -> float:
    """Plot and save ROC curve; return AUC."""
    os.makedirs(output_dir, exist_ok=True)
    from sklearn.metrics import roc_auc_score, roc_curve

    fpr, tpr, _ = roc_curve(y_true, calibrated_pds)
    auc = float(roc_auc_score(y_true, calibrated_pds))
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, color="#7F77DD", linewidth=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], color="#888780", linestyle="--", linewidth=1, label="Random")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve — FULL tier (test set)")
    ax.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "roc_curve.png"), dpi=150)
    plt.close()
    return auc


def compute_fbeta(
    y_true: np.ndarray,
    y_pred_binary: np.ndarray,
    beta: float = 5.0,
) -> float:
    """Compute F-beta score."""
    from sklearn.metrics import fbeta_score

    return float(fbeta_score(y_true, y_pred_binary, beta=beta, zero_division=0))


def plot_fbeta_sweep(
    y_true: np.ndarray,
    calibrated_pds: np.ndarray,
    beta: float = 5.0,
    output_dir: str = EVAL_PLOTS_DIR,
) -> float:
    """Sweep thresholds 0.01–0.99 for F-beta; return best threshold."""
    os.makedirs(output_dir, exist_ok=True)
    thresholds = np.arange(0.01, 1.00, 0.01)
    fbeta_scores = []
    for t in thresholds:
        y_pred = (calibrated_pds >= t).astype(int)
        fbeta_scores.append(compute_fbeta(y_true, y_pred, beta))
    fbeta_scores = np.array(fbeta_scores)
    best_idx = int(np.argmax(fbeta_scores))
    best_threshold = float(thresholds[best_idx])
    best_fbeta = float(fbeta_scores[best_idx])

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(thresholds, fbeta_scores, color="#D85A30", linewidth=2)
    ax.axvline(
        x=best_threshold,
        color="#1D9E75",
        linestyle="--",
        linewidth=1.5,
        label=f"Best threshold={best_threshold:.2f} (F{beta:.0f}={best_fbeta:.3f})",
    )
    ax.axvline(x=0.15, color="#888780", linestyle=":", linewidth=1, label="APPROVE boundary (0.15)")
    ax.axvline(x=0.35, color="#888780", linestyle=":", linewidth=1, label="DECLINE boundary (0.35)")
    ax.set_xlabel("Threshold")
    ax.set_ylabel(f"F-beta (\u03b2={beta:.0f})")
    ax.set_title(
        f"F-beta(\u03b2={beta:.0f}) sweep — diagnostic only\n"
        "F-beta(\u03b2=5) is a recall-weighted operating diagnostic. "
        "The production policy is a fixed three-band business rule."
    )
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fbeta_sweep.png"), dpi=150)
    plt.close()
    return best_threshold


# ──────────────────────────────────────────────────────────────────────
# Entry point — unit tests
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Stage 1 — Reason Mapping & top_5_explanations Unit Tests")
    print("=" * 60)

    # render_reason tests
    assert render_reason("CREDIT_INCOME_RATIO") == \
        "Requested credit is high relative to stated income"
    assert render_reason("BUREAU_LOAN_COUNT") == \
        "External credit history indicates elevated repayment risk"
    assert render_reason("CC_DPD_MAX") == \
        "Credit card utilization or delinquency history indicates elevated risk"
    assert render_reason("POS_RECORD_COUNT") == \
        "Past point-of-sale loan behavior indicates elevated risk"
    assert render_reason("INST_DPD_MEAN") == \
        "Past installment payments show late-payment behavior"
    assert render_reason("AGE_YEARS") == \
        "Combined application and repayment profile increased model risk"
    assert render_reason("bureau_loan_count") == \
        "Combined application and repayment profile increased model risk"
    print("  render_reason ................ PASSED")

    # top_5_explanations_from_shap tests
    test_series = pd.Series({
        "BUREAU_LOAN_COUNT":   0.42,
        "CREDIT_INCOME_RATIO": -0.31,
        "AGE_YEARS":           0.18,
        "CC_DPD_MAX":          -0.55,
        "EXT_SOURCE_MEAN":     0.67,
        "POS_RECORD_COUNT":    -0.11,
    })
    result = top_5_explanations_from_shap(test_series)
    assert len(result) == 5
    assert result[0]["feature"] == "EXT_SOURCE_MEAN"
    assert isinstance(result[0]["reason"], str)
    assert all("feature" in r and "reason" in r for r in result)
    print("  top_5_explanations_from_shap . PASSED")

    # ValueError on empty series
    try:
        top_5_explanations_from_shap(pd.Series(dtype=float))
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    print("  ValueError on empty series ... PASSED")
    print("Stage 1 unit tests passed\n")

    # ── Stage 2 — SHAP plots ────────────────────────────────────────
    print("=" * 60)
    print("Stage 2 — SHAP Plot Generation")
    print("=" * 60)

    import xgboost as xgb
    import shap

    rng = np.random.default_rng(42)
    X_synth = rng.random((300, 20)).astype(np.float32)
    y_synth = rng.integers(0, 2, 300)
    clf = xgb.XGBClassifier(n_estimators=50, random_state=42, eval_metric="logloss")
    clf.fit(X_synth, y_synth)
    explainer = shap.TreeExplainer(clf)
    feature_names = [f"feat_{i}" for i in range(20)]
    X_test_synth = rng.random((100, 20)).astype(np.float32)
    cal_pds = clf.predict_proba(X_test_synth)[:, 1]

    generate_shap_plots(
        explainer=explainer,
        X_test=X_test_synth,
        feature_names=feature_names,
        calibrated_pds=cal_pds,
        output_dir=SHAP_PLOTS_DIR,
    )

    for fname in [
        "global_summary_bar.png", "beeswarm.png", "heatmap.png",
        "local_approve.png", "local_review.png", "local_decline.png",
    ]:
        path = os.path.join(SHAP_PLOTS_DIR, fname)
        assert os.path.exists(path), f"Missing: {path}"
        assert os.path.getsize(path) > 5000, f"Too small: {path}"
    print("Stage 2 synthetic SHAP plot tests passed\n")

    # ── Stage 3 — Eval plots ────────────────────────────────────────
    print("=" * 60)
    print("Stage 3 — Eval Plots")
    print("=" * 60)

    rng2 = np.random.default_rng(99)
    y_true_synth = rng2.integers(0, 2, 300)
    pds_synth = rng2.uniform(0, 1, 300)

    plot_confusion_matrix(y_true_synth, pds_synth, threshold=0.35, output_dir=EVAL_PLOTS_DIR)
    auc = plot_roc_curve(y_true_synth, pds_synth, output_dir=EVAL_PLOTS_DIR)
    assert isinstance(auc, float)
    assert 0.0 <= auc <= 1.0

    best_t = plot_fbeta_sweep(y_true_synth, pds_synth, beta=5.0, output_dir=EVAL_PLOTS_DIR)
    assert 0.01 <= best_t <= 0.99

    for fname in ["confusion_matrix.png", "roc_curve.png", "fbeta_sweep.png"]:
        path = os.path.join(EVAL_PLOTS_DIR, fname)
        assert os.path.exists(path), f"Missing: {path}"
        assert os.path.getsize(path) > 5000
    print("Stage 3 eval plot tests passed\n")

    print("=" * 60)
    print("src/explainability.py — ALL STAGES PASSED")
    print("=" * 60)
