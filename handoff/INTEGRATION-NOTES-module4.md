## Module 4 â€” Explainability & Fairness: Integration Notes

### What this module does
Generates 6 SHAP plots (global summary bar, beeswarm, heatmap, 3 local force plots),
3 evaluation plots (confusion matrix, ROC curve, F-beta sweep), runs a proxy fairness
audit across 3 group families with 3 metrics each, produces 3 audit CSVs and a fairness
summary card PNG, and overwrites `artifacts/model_fairness_audit_passed.joblib` with the
real boolean result.

### What it expects from other modules
- **Module 1**: `data/processed/train.pkl`, `val_policy.pkl`, `test.pkl`; train
  `AMT_INCOME_TOTAL` for quantile computation
- **Module 2**: `build_full()` / `FrozenFeatureBuilder` for feature matrix construction;
  `pool_rare_categories()` used inside `derive_fairness_groups`
- **Module 3**: `artifacts/full_model.joblib`, `full_calibrator.joblib`,
  `full_shap_explainer.joblib`; `decision_from_pd()` for DECISION column construction

### What other modules can call from it
```python
from src.explainability import render_reason, top_5_explanations_from_shap
from src.fairness_audit import run_full_fairness_audit
```

Module 5 imports `render_reason` and `top_5_explanations_from_shap` for the `/score`
endpoint.

### Known edge cases the integration AI must be aware of
1. `model_fairness_audit_passed.joblib` is overwritten by this module. Module 5 must
   load it AFTER Module 4 has run. Load order in integration: M3 â†’ M4 â†’ M5.
2. The fairness audit DataFrame passed to `run_full_fairness_audit()` must contain
   columns: `TARGET`, `CALIBRATED_PD`, `DECISION`, `AMT_INCOME_TOTAL`,
   `REGION_RATING_CLIENT_W_CITY`, `NAME_INCOME_TYPE`, `NAME_HOUSING_TYPE`,
   `FLAG_OWN_CAR`, `FLAG_OWN_REALTY`, `CNT_CHILDREN`. These are raw columns from the
   split DataFrame â€” NOT from `build_full()` output (which drops fairness columns).
3. With only 200 rows in synthetic mode, most fairness cells will be ineligible
   (n < 200). This is expected â€” the audit returns `False`. Real data (~61k test rows)
   will have sufficient cell sizes.
4. `top_5_explanations_from_shap` expects a `pd.Series` indexed by feature name â€” not a
   raw numpy array. Module 5 must build this Series from SHAP values.
5. `plot_fbeta_sweep` returns the best diagnostic threshold as a float. This is NOT used
   to set the decision policy. The production policy uses fixed thresholds 0.15/0.35.
6. The existing Module 2 uses a `FrozenFeatureBuilder` pattern. The `main()` function in
   `fairness_audit.py` handles both loading a saved builder and fitting a new one.


