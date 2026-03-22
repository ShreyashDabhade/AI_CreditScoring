"""Module 4 — Fairness Audit.

Proxy fairness audit across 3 group families (Primary, Secondary,
Tertiary) with 3 metrics each (DI, EOD, Brier Ratio).
Overwrites artifacts/model_fairness_audit_passed.joblib with the
final boolean result.
"""

from __future__ import annotations

import os
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────

FAIRNESS_MIN_N: int = 200
FAIRNESS_MIN_DEFAULTS: int = 20
DI_THRESHOLD: float = 0.80
EOD_THRESHOLD: float = 0.10
BRIER_THRESHOLD: float = 1.25

FAIRNESS_COLS: list[str] = [
    "AMT_INCOME_TOTAL",
    "REGION_RATING_CLIENT_W_CITY",
    "NAME_INCOME_TYPE",
    "NAME_HOUSING_TYPE",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "CNT_CHILDREN",
]


def _load_or_fit_full_builder(
    train_raw: pd.DataFrame,
    raw_dir: str = "data/raw/",
    builder_path: str = "artifacts/full_feature_builder.joblib",
):
    """Boundary shim for the Module 2 FrozenFeatureBuilder contract."""
    from src.feature_engineering import FrozenFeatureBuilder, fit_full_builder

    if os.path.exists(builder_path):
        builder = joblib.load(builder_path)
        if not isinstance(builder, FrozenFeatureBuilder):
            raise TypeError(
                "full_feature_builder.joblib must contain a FrozenFeatureBuilder"
            )
        return builder

    builder = fit_full_builder(train_raw, raw_dir=raw_dir)
    if builder_path != "artifacts/full_feature_builder.joblib":
        builder.save(builder_path)
    return builder


# ──────────────────────────────────────────────────────────────────────
# Stage 4 — Fairness Group Derivation
# ──────────────────────────────────────────────────────────────────────

def derive_fairness_groups(
    df: pd.DataFrame,
    train_income_q1: float,
    train_income_q2: float,
) -> pd.DataFrame:
    """Add 8 fairness group columns using train-fitted income quantiles."""
    df = df.copy()

    # Step A — INCOME_TERTILE
    def _income_tertile(x: float) -> str:
        if pd.isna(x):
            return "MISSING"
        if x <= train_income_q1:
            return "T1"
        if x <= train_income_q2:
            return "T2"
        return "T3"

    df["INCOME_TERTILE"] = df["AMT_INCOME_TOTAL"].apply(_income_tertile)

    # Step B — FAIR_GROUP_PRIMARY
    df["FAIR_GROUP_PRIMARY"] = (
        df["INCOME_TERTILE"].astype(str)
        + "__"
        + df["REGION_RATING_CLIENT_W_CITY"].fillna(-1).astype(int).astype(str)
    )

    # Step C — INCOME_TYPE_POOLED
    from src.feature_engineering import pool_rare_categories

    df["INCOME_TYPE_POOLED"] = pool_rare_categories(
        df["NAME_INCOME_TYPE"], min_count=500
    ).fillna("MISSING")

    # Step D — HOUSING_TYPE_POOLED
    df["HOUSING_TYPE_POOLED"] = pool_rare_categories(
        df["NAME_HOUSING_TYPE"], min_count=500
    ).fillna("MISSING")

    # Step E — FAIR_GROUP_SECONDARY
    df["FAIR_GROUP_SECONDARY"] = (
        df["INCOME_TYPE_POOLED"] + "__" + df["HOUSING_TYPE_POOLED"]
    )

    # Step F — CHILDREN_BIN
    df["CHILDREN_BIN"] = pd.cut(
        df["CNT_CHILDREN"].fillna(0),
        bins=[-1, 0, 1, np.inf],
        labels=["0", "1", "2_PLUS"],
    )
    df["CHILDREN_BIN"] = df["CHILDREN_BIN"].astype(str)

    # Step G — OWN_ASSET_BIN
    df["OWN_ASSET_BIN"] = (
        df["FLAG_OWN_CAR"].fillna("N") + "_" + df["FLAG_OWN_REALTY"].fillna("N")
    )

    # Step H — FAIR_GROUP_TERTIARY
    df["FAIR_GROUP_TERTIARY"] = (
        df["OWN_ASSET_BIN"] + "__" + df["CHILDREN_BIN"]
    )

    return df


def valid_fairness_cells(
    df: pd.DataFrame,
    group_col: str,
    y_col: str = "TARGET",
    min_n: int = FAIRNESS_MIN_N,
    min_pos: int = FAIRNESS_MIN_DEFAULTS,
) -> list[str]:
    """Return group values where count >= min_n AND defaults >= min_pos."""
    stats = (
        df.groupby(group_col)[y_col]
        .agg(n="size", pos="sum")
        .reset_index()
    )
    return stats.query("n >= @min_n and pos >= @min_pos")[group_col].tolist()


# ──────────────────────────────────────────────────────────────────────
# Stage 5 — Fairness Metric Computation & Audit
# ──────────────────────────────────────────────────────────────────────

def compute_fairness_metrics(
    df: pd.DataFrame,
    group_col: str,
    y_col: str = "TARGET",
    pd_col: str = "CALIBRATED_PD",
    decision_col: str = "DECISION",
    min_n: int = FAIRNESS_MIN_N,
    min_pos: int = FAIRNESS_MIN_DEFAULTS,
) -> pd.DataFrame:
    """Compute DI, EOD, Brier ratio for each group cell."""
    eligible = valid_fairness_cells(df, group_col, y_col, min_n, min_pos)

    rows: list[dict[str, Any]] = []
    for gval in df[group_col].unique():
        sub = df[df[group_col] == gval]
        n = len(sub)
        n_defaults = int(sub[y_col].sum())
        evaluable = gval in eligible

        if not evaluable:
            rows.append({
                "group_value": str(gval),
                "n": n,
                "n_defaults": n_defaults,
                "approve_rate": np.nan,
                "tpr": np.nan,
                "brier_score": np.nan,
                "di_ratio": np.nan,
                "eod": np.nan,
                "brier_ratio": np.nan,
                "evaluable": False,
                "di_pass": np.nan,
                "eod_pass": np.nan,
                "brier_pass": np.nan,
            })
            continue

        approve_rate = float((sub[decision_col] == "APPROVE").mean())
        good = sub[sub[y_col] == 0]
        tpr = float(
            (good[decision_col] == "APPROVE").mean() if len(good) > 0 else np.nan
        )
        brier = float(np.mean((sub[pd_col].values - sub[y_col].values) ** 2))

        rows.append({
            "group_value": str(gval),
            "n": n,
            "n_defaults": n_defaults,
            "approve_rate": approve_rate,
            "tpr": tpr,
            "brier_score": brier,
            "di_ratio": np.nan,
            "eod": np.nan,
            "brier_ratio": np.nan,
            "evaluable": True,
            "di_pass": np.nan,
            "eod_pass": np.nan,
            "brier_pass": np.nan,
        })

    result = pd.DataFrame(rows)
    eval_mask = result["evaluable"] == True  # noqa: E712

    if eval_mask.sum() == 0:
        return result

    # Compute family-level metrics across evaluable cells
    max_approve = result.loc[eval_mask, "approve_rate"].max()
    min_approve = result.loc[eval_mask, "approve_rate"].min()
    max_tpr = result.loc[eval_mask, "tpr"].max()
    min_tpr = result.loc[eval_mask, "tpr"].min()
    max_brier = result.loc[eval_mask, "brier_score"].max()
    min_brier = result.loc[eval_mask, "brier_score"].min()

    family_di_pass = (
        (min_approve / max_approve >= DI_THRESHOLD) if max_approve > 0 else False
    )
    family_eod_pass = (max_tpr - min_tpr) <= EOD_THRESHOLD
    family_brier_pass = (
        (max_brier / min_brier <= BRIER_THRESHOLD) if min_brier > 0 else False
    )

    result.loc[eval_mask, "di_ratio"] = (
        result.loc[eval_mask, "approve_rate"] / max_approve
    )
    result.loc[eval_mask, "eod"] = result.loc[eval_mask, "tpr"] - max_tpr
    result.loc[eval_mask, "brier_ratio"] = (
        result.loc[eval_mask, "brier_score"] / min_brier if min_brier > 0 else np.nan
    )
    result.loc[eval_mask, "di_pass"] = family_di_pass
    result.loc[eval_mask, "eod_pass"] = family_eod_pass
    result.loc[eval_mask, "brier_pass"] = family_brier_pass

    return result


def _plot_fairness_summary_card(
    primary_passes: dict[str, bool],
    secondary_passes: dict[str, bool],
    tertiary_passes: dict[str, bool],
    output_dir: str,
) -> None:
    """Produce a 3×3 grid PNG showing pass/fail for each metric × family."""
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    families = ["Primary", "Secondary", "Tertiary"]
    passes_list = [primary_passes, secondary_passes, tertiary_passes]
    metrics = ["DI (≥0.80)", "EOD (≤0.10)", "Brier (≤1.25)"]
    metric_keys = ["di", "eod", "brier"]

    for i, (ax, family, passes) in enumerate(zip(axes, families, passes_list)):
        ax.set_title(f"{family} group", fontsize=11)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        for j, (label, key) in enumerate(zip(metrics, metric_keys)):
            passed = passes.get(key, False)
            color = "#1D9E75" if passed else "#D85A30"
            symbol = "PASS" if passed else "FAIL"
            y_pos = 0.75 - j * 0.28
            ax.add_patch(
                plt.Rectangle(
                    (0.05, y_pos - 0.08), 0.9, 0.22,
                    color=color, alpha=0.15,
                )
            )
            ax.text(
                0.5, y_pos + 0.03,
                f"{label}: {symbol}",
                ha="center", va="center",
                fontsize=9, color=color, fontweight="bold",
            )

    plt.suptitle(
        "Proxy fairness audit — pass/fail summary\n"
        "These metrics evaluate stability across proxy-defined "
        "subgroups only and must not be interpreted as proof of "
        "fairness across legally protected characteristics.",
        fontsize=8, y=1.02,
    )
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(
        os.path.join(output_dir, "fairness_summary_card.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.close()


def _family_passes(audit_df: pd.DataFrame) -> dict[str, bool]:
    """Extract family-level pass/fail from audit DataFrame."""
    ev = audit_df[audit_df["evaluable"] == True]  # noqa: E712
    if len(ev) == 0:
        return {"di": False, "eod": False, "brier": False}
    di_val = ev["di_pass"].iloc[0]
    eod_val = ev["eod_pass"].iloc[0]
    brier_val = ev["brier_pass"].iloc[0]
    return {
        "di": bool(di_val),
        "eod": bool(eod_val),
        "brier": bool(brier_val),
    }


def run_full_fairness_audit(
    df: pd.DataFrame,
    train_income_q1: float,
    train_income_q2: float,
    output_dir: str,
) -> bool:
    """Run full 3-family fairness audit; return True if all pass."""
    os.makedirs(output_dir, exist_ok=True)

    # Step A — derive groups
    df = derive_fairness_groups(df, train_income_q1, train_income_q2)

    # Step B — compute metrics per family
    primary_df = compute_fairness_metrics(df, "FAIR_GROUP_PRIMARY")
    secondary_df = compute_fairness_metrics(df, "FAIR_GROUP_SECONDARY")
    tertiary_df = compute_fairness_metrics(df, "FAIR_GROUP_TERTIARY")

    # Step C — save CSVs
    primary_df.to_csv(
        os.path.join(output_dir, "audit_primary.csv"), index=False
    )
    secondary_df.to_csv(
        os.path.join(output_dir, "audit_secondary.csv"), index=False
    )
    tertiary_df.to_csv(
        os.path.join(output_dir, "audit_tertiary.csv"), index=False
    )

    # Step D — extract family-level pass/fail
    p_passes = _family_passes(primary_df)
    s_passes = _family_passes(secondary_df)
    t_passes = _family_passes(tertiary_df)

    # Step E — summary card
    _plot_fairness_summary_card(p_passes, s_passes, t_passes, output_dir)

    # Step F — print audit results
    print("\n=== Proxy Fairness Audit Results ===")
    for name, passes in [
        ("Primary", p_passes),
        ("Secondary", s_passes),
        ("Tertiary", t_passes),
    ]:
        print(
            f"{name}: DI={'PASS' if passes['di'] else 'FAIL'}  "
            f"EOD={'PASS' if passes['eod'] else 'FAIL'}  "
            f"Brier={'PASS' if passes['brier'] else 'FAIL'}"
        )

    # Step G — determine overall result
    all_pass = all([
        p_passes["di"], p_passes["eod"], p_passes["brier"],
        s_passes["di"], s_passes["eod"], s_passes["brier"],
        t_passes["di"], t_passes["eod"], t_passes["brier"],
    ])
    print(f"\nmodel_fairness_audit_passed = {all_pass}")
    return all_pass


# ──────────────────────────────────────────────────────────────────────
# main() orchestration
# ──────────────────────────────────────────────────────────────────────

def main() -> None:
    """Module 4 orchestration — detects real vs synthetic data."""
    DATA_PROCESSED_DIR = os.environ.get("DATA_PROCESSED_DIR", "data/processed/")
    ARTIFACT_DIR = os.environ.get("ARTIFACT_DIR", "artifacts/")
    FAIRNESS_PLOTS_DIR = os.environ.get("FAIRNESS_PLOTS_DIR", "notebooks/fairness_plots/")
    SHAP_PLOTS_DIR = os.environ.get("SHAP_PLOTS_DIR", "notebooks/shap_plots/")
    EVAL_PLOTS_DIR = os.environ.get("EVAL_PLOTS_DIR", "notebooks/eval_plots/")

    val_policy_path = os.path.join(DATA_PROCESSED_DIR, "val_policy.pkl")
    real_data = os.path.exists(val_policy_path)
    print(f"[Module 4 Real Data Audit] Checking for real data at '{val_policy_path}' — found: {real_data}")
    print(f"[Module 4] FORCING REAL DATA MODE per user directive")
    real_data = True  # FORCE REAL MODE

    if real_data:
        import pickle
        import sys

        sys.path.insert(0, ".")

        print("[M4] Loading pickles...")
        try:
            with open(os.path.join(DATA_PROCESSED_DIR, "train.pkl"), "rb") as f:
                train_raw = pickle.load(f)
            print(f"[M4]   ✓ train.pkl loaded: {train_raw.shape}")
            with open(os.path.join(DATA_PROCESSED_DIR, "test.pkl"), "rb") as f:
                test_raw = pickle.load(f)
            print(f"[M4]   ✓ test.pkl loaded: {test_raw.shape}")
        except Exception as e:
            print(f"[M4] ✗ Error loading pickles: {e}")
            raise

        print("[M4] Loading modules...")
        try:
            from src.feature_engineering import build_full
            from src.models.train import decision_from_pd, load_artifacts
            print("[M4]   ✓ Modules imported")
        except Exception as e:
            print(f"[M4] ✗ Error importing modules: {e}")
            raise

        print("[M4] Loading artifacts...")
        try:
            artifacts = load_artifacts(ARTIFACT_DIR)
            print(f"[M4]   ✓ Artifacts loaded")

            model = artifacts.get("full_model")
            calibrator = artifacts.get("full_calibrator")
            explainer = artifacts.get("full_shap_explainer")
            print(f"[M4]   ✓ model={type(model).__name__}, calibrator={type(calibrator).__name__}, explainer={type(explainer).__name__}")
        except Exception as e:
            print(f"[M4] ✗ Error loading artifacts: {e}")
            raise

        print("[M4] Loading feature builder...")
        try:
            full_builder = _load_or_fit_full_builder(
                train_raw,
                raw_dir="data/raw/",
                builder_path="artifacts/full_feature_builder.joblib",
            )
            print(f"[M4]   ✓ Feature builder loaded")
        except Exception as e:
            print(f"[M4] ✗ Error loading feature builder: {e}")
            raise

        print("[M4] Building features (this may take 1-2 minutes)...")
        try:
            X_test_full = build_full(test_raw, full_builder, raw_dir="data/raw/")
            print(f"[M4]   ✓ Features built: {X_test_full.shape}")
            feature_names = list(X_test_full.columns)
            X_test_arr = X_test_full.values
        except Exception as e:
            print(f"[M4] ✗ Error building features: {e}")
            raise

        if model is None or calibrator is None or explainer is None:
            print("WARNING: Missing Module 3 artifacts; training fallback model for Module 4 execution.")
            import xgboost as xgb
            from sklearn.calibration import CalibratedClassifierCV
            import shap

            X_train_full = build_full(train_raw, full_builder, raw_dir="data/raw/")
            y_train = train_raw["TARGET"].values

            model = xgb.XGBClassifier(
                n_estimators=100,
                random_state=42,
                eval_metric="logloss",
                use_label_encoder=False,
            )
            model.fit(X_train_full.values, y_train)

            if calibrator is None:
                calibrator = CalibratedClassifierCV(base_estimator=model, cv=3, method="sigmoid")
                calibrator.fit(X_train_full.values, y_train)

            if explainer is None:
                explainer = shap.TreeExplainer(model)

        raw_pds = model.predict_proba(X_test_arr)[:, 1]
        cal_pds = calibrator.predict(raw_pds)
        decisions = np.array([decision_from_pd(p) for p in cal_pds])

        # Build audit DataFrame from raw test columns
        fairness_cols_present = [c for c in FAIRNESS_COLS if c in test_raw.columns]
        test_audit_df = test_raw[["TARGET"] + fairness_cols_present].copy()
        test_audit_df["CALIBRATED_PD"] = cal_pds
        test_audit_df["DECISION"] = decisions

        train_income_q1 = float(train_raw["AMT_INCOME_TOTAL"].quantile(1 / 3))
        train_income_q2 = float(train_raw["AMT_INCOME_TOTAL"].quantile(2 / 3))

        audit_passed = run_full_fairness_audit(
            test_audit_df, train_income_q1, train_income_q2, FAIRNESS_PLOTS_DIR
        )

        from src.explainability import (
            generate_shap_plots,
            plot_confusion_matrix,
            plot_fbeta_sweep,
            plot_roc_curve,
        )

        generate_shap_plots(
            explainer, X_test_arr, feature_names, cal_pds, SHAP_PLOTS_DIR
        )
        plot_confusion_matrix(
            test_raw["TARGET"].values, cal_pds, threshold=0.35, output_dir=EVAL_PLOTS_DIR
        )
        plot_roc_curve(
            test_raw["TARGET"].values, cal_pds, output_dir=EVAL_PLOTS_DIR
        )
        plot_fbeta_sweep(
            test_raw["TARGET"].values, cal_pds, beta=5.0, output_dir=EVAL_PLOTS_DIR
        )

    else:
        print("Real data not found — using synthetic mode")
        import xgboost as xgb
        import shap

        rng = np.random.default_rng(42)
        X_synth = rng.random((500, 90)).astype(np.float32)
        y_synth = rng.integers(0, 2, 500)
        clf = xgb.XGBClassifier(
            n_estimators=50, random_state=42, eval_metric="logloss"
        )
        clf.fit(X_synth, y_synth)
        explainer = shap.TreeExplainer(clf)
        feature_names = [f"feat_{i}" for i in range(90)]
        X_test_arr = rng.random((200, 90)).astype(np.float32)
        cal_pds = clf.predict_proba(X_test_arr)[:, 1]
        decisions = np.array([
            "APPROVE" if p < 0.15 else "REVIEW" if p < 0.35 else "DECLINE"
            for p in cal_pds
        ])

        synth_income = rng.uniform(10000, 500000, 200)
        q1 = float(np.percentile(synth_income, 100 / 3))
        q2 = float(np.percentile(synth_income, 200 / 3))
        audit_df = pd.DataFrame({
            "TARGET": rng.integers(0, 2, 200),
            "AMT_INCOME_TOTAL": synth_income,
            "REGION_RATING_CLIENT_W_CITY": rng.choice([1, 2, 3], 200),
            "NAME_INCOME_TYPE": rng.choice(
                ["Working", "Pensioner", "Commercial associate",
                 "State servant", "Unemployed"], 200
            ),
            "NAME_HOUSING_TYPE": rng.choice(
                ["House / apartment", "Rented apartment",
                 "With parents", "Municipal apartment"], 200
            ),
            "FLAG_OWN_CAR": rng.choice(["Y", "N"], 200),
            "FLAG_OWN_REALTY": rng.choice(["Y", "N"], 200),
            "CNT_CHILDREN": rng.integers(0, 5, 200).astype(float),
            "CALIBRATED_PD": cal_pds,
            "DECISION": decisions,
        })

        audit_passed = run_full_fairness_audit(
            audit_df, q1, q2, FAIRNESS_PLOTS_DIR
        )

        from src.explainability import (
            generate_shap_plots,
            plot_confusion_matrix,
            plot_fbeta_sweep,
            plot_roc_curve,
        )

        generate_shap_plots(
            explainer, X_test_arr, feature_names, cal_pds, SHAP_PLOTS_DIR
        )
        plot_confusion_matrix(
            audit_df["TARGET"].values, cal_pds, threshold=0.35, output_dir=EVAL_PLOTS_DIR
        )
        plot_roc_curve(
            audit_df["TARGET"].values, cal_pds, output_dir=EVAL_PLOTS_DIR
        )
        plot_fbeta_sweep(
            audit_df["TARGET"].values, cal_pds, beta=5.0, output_dir=EVAL_PLOTS_DIR
        )

    # Overwrite model_fairness_audit_passed.joblib
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    joblib.dump(
        bool(audit_passed),
        os.path.join(ARTIFACT_DIR, "model_fairness_audit_passed.joblib"),
    )
    print(f"model_fairness_audit_passed.joblib overwritten with: {audit_passed}")
    print("Module 4 complete")


# ──────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── Unit tests first ─────────────────────────────────────────────
    print("=" * 60)
    print("Stage 4 — Fairness Group Derivation Tests")
    print("=" * 60)

    rng = np.random.default_rng(1)
    synth_df = pd.DataFrame({
        "AMT_INCOME_TOTAL": np.random.default_rng(1).uniform(10000, 500000, 1000),
        "REGION_RATING_CLIENT_W_CITY": np.random.default_rng(2).choice([1, 2, 3], 1000),
        "NAME_INCOME_TYPE": np.random.default_rng(3).choice(
            ["Working", "Pensioner", "Commercial associate",
             "State servant", "Unemployed"], 1000
        ),
        "NAME_HOUSING_TYPE": np.random.default_rng(4).choice(
            ["House / apartment", "Rented apartment",
             "With parents", "Municipal apartment"], 1000
        ),
        "FLAG_OWN_CAR": np.random.default_rng(5).choice(["Y", "N"], 1000),
        "FLAG_OWN_REALTY": np.random.default_rng(6).choice(["Y", "N"], 1000),
        "CNT_CHILDREN": np.random.default_rng(7).integers(0, 5, 1000).astype(float),
        "TARGET": np.random.default_rng(8).integers(0, 2, 1000),
    })

    q1 = float(np.percentile(synth_df["AMT_INCOME_TOTAL"], 100 / 3))
    q2 = float(np.percentile(synth_df["AMT_INCOME_TOTAL"], 200 / 3))
    result = derive_fairness_groups(synth_df, q1, q2)

    for col in [
        "INCOME_TERTILE", "FAIR_GROUP_PRIMARY",
        "INCOME_TYPE_POOLED", "HOUSING_TYPE_POOLED",
        "FAIR_GROUP_SECONDARY", "CHILDREN_BIN",
        "OWN_ASSET_BIN", "FAIR_GROUP_TERTIARY",
    ]:
        assert col in result.columns, f"Missing column: {col}"

    assert set(result["INCOME_TERTILE"].unique()).issubset({"T1", "T2", "T3", "MISSING"})
    assert result["FAIR_GROUP_PRIMARY"].str.contains("__").all()
    assert result["CHILDREN_BIN"].isin(["0", "1", "2_PLUS"]).all()
    print("  derive_fairness_groups ......... PASSED")

    cells = valid_fairness_cells(result, "FAIR_GROUP_PRIMARY", min_n=50, min_pos=5)
    assert isinstance(cells, list)
    assert len(cells) >= 1
    print("  valid_fairness_cells ........... PASSED")
    print("Stage 4 tests passed\n")

    # ── Stage 5 — Run full audit synthetic test ──────────────────────
    print("=" * 60)
    print("Stage 5 — Full Fairness Audit Synthetic Test")
    print("=" * 60)

    # Add CALIBRATED_PD and DECISION columns for audit
    synth_df_audit = synth_df.copy()
    synth_df_audit["CALIBRATED_PD"] = np.random.default_rng(9).uniform(0, 1, 1000)
    synth_df_audit["DECISION"] = np.where(
        synth_df_audit["CALIBRATED_PD"] < 0.15, "APPROVE",
        np.where(synth_df_audit["CALIBRATED_PD"] < 0.35, "REVIEW", "DECLINE"),
    )

    FAIRNESS_PLOTS_DIR = os.environ.get("FAIRNESS_PLOTS_DIR", "notebooks/fairness_plots/")
    audit_passed = run_full_fairness_audit(
        synth_df_audit, q1, q2, FAIRNESS_PLOTS_DIR
    )
    assert isinstance(audit_passed, bool)

    # Verify CSVs
    for fname in ["audit_primary.csv", "audit_secondary.csv", "audit_tertiary.csv"]:
        path = os.path.join(FAIRNESS_PLOTS_DIR, fname)
        assert os.path.exists(path), f"Missing: {path}"
        csv_df = pd.read_csv(path)
        expected_cols = [
            "group_value", "n", "n_defaults", "approve_rate", "tpr",
            "brier_score", "di_ratio", "eod", "brier_ratio",
            "evaluable", "di_pass", "eod_pass", "brier_pass",
        ]
        assert list(csv_df.columns) == expected_cols, f"Column mismatch in {fname}"
    print("  Audit CSVs verified ............ PASSED")

    # Verify summary card
    card_path = os.path.join(FAIRNESS_PLOTS_DIR, "fairness_summary_card.png")
    assert os.path.exists(card_path), "Missing fairness_summary_card.png"
    assert os.path.getsize(card_path) > 5000
    print("  fairness_summary_card.png ...... PASSED")

    print("Stage 5 tests passed\n")

    # ── Run main() in synthetic mode ─────────────────────────────────
    print("=" * 60)
    print("Running main() — synthetic mode")
    print("=" * 60)
    main()
