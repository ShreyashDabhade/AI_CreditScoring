## Module 3 - Model Training: Integration Notes

### What this module does
- fits and persists lineage-validated FULL and REDUCED builders into the chosen artifact directory
- trains a FULL weighted XGBoost+LightGBM blend runtime candidate plus persisted FULL XGBoost fallback artifacts
- trains a REDUCED XGBoost tier
- selects the deployed FULL runtime candidate on `val_model`
- calibrates probabilities on `val_policy`
- persists models, calibrators, explainers, weighted-blend metadata, and the reproducibility report into the chosen artifact directory
- writes `model_fairness_audit_passed.joblib` as placeholder `False`

### Runtime contract
- `load_artifacts()` scopes every load to the passed `artifact_dir`
- builders must be loaded through `src/builder_artifacts.py`
- the reproducibility report must describe only what the current source actually trains

### Truthfulness rule
- FULL runtime truth must remain aligned with the current deployed weighted-blend candidate and its persisted fallback XGBoost artifacts.
- REDUCED runtime truth must remain aligned with the persisted REDUCED XGBoost tier.
- Offline experiment entrypoints may evaluate additional candidates, but those experiments must not be described as the default live runtime unless they are actually deployed.

### Fairness ownership
Module 3 never writes a final fairness result. Module 4 is the sole owner of the final overwrite.
