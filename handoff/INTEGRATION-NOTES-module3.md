## Module 3 - Model Training: Integration Notes

### What this module does
- trains FULL and REDUCED XGBoost candidate grids
- selects the best candidate on `val_model`
- calibrates probabilities on `val_policy`
- persists builders, models, calibrators, explainers, and the reproducibility report into the chosen artifact directory
- writes `model_fairness_audit_passed.joblib` as placeholder `False`

### Runtime contract
- `load_artifacts()` scopes every load to the passed `artifact_dir`
- builders must be loaded through `src/builder_artifacts.py`
- the reproducibility report must describe only what the current source actually trains

### Truthfulness rule
If this branch trains only XGBoost candidates, the report and handoff must not claim Logistic Regression or LightGBM evaluation.

### Fairness ownership
Module 3 never writes a final fairness result. Module 4 is the sole owner of the final overwrite.
