# Backend Freeze Summary

## Active Runtime Model And Config

- Default processed runtime path: `data/processed/`
- Default artifact runtime path: `artifacts/`
- Supported split modes: `proxy_time`, `random_stratified`
- Current default split mode: `proxy_time`
- Runtime API startup remains strict and fail-fast in real mode

Current runtime model versions from the canonical artifact stack:

- FULL: `full_weighted_blend_v2.2.0`
- REDUCED: `reduced_v2.1.0`

## Runtime Artifacts Used By The API

Processed lineage:

- `data/processed/processed_artifact_manifest.json`

Builders:

- `artifacts/full_feature_builder.joblib`
- `artifacts/full_feature_builder.manifest.json`
- `artifacts/reduced_feature_builder.joblib`
- `artifacts/reduced_feature_builder.manifest.json`

Scoring stack:

- `artifacts/full_model.joblib`
- `artifacts/full_calibrator.joblib`
- `artifacts/full_shap_explainer.joblib`
- `artifacts/reduced_model.joblib`
- `artifacts/reduced_calibrator.joblib`
- `artifacts/reduced_shap_explainer.joblib`
- `artifacts/model_fairness_audit_passed.joblib`
- optional `artifacts/reproducibility_report.json`

## Preserved But Not Active

These are preserved as offline evidence and are not runtime API inputs:

- `artifacts/reduced_thin_blend/`
- `artifacts/reduced_thin_lgbm/`
- `artifacts/reduced_thin_diag/`
- `artifacts/reduced_thin_opt/`
- `artifacts/alt_stacked_reduced/`
- `artifacts/random_stratified_reduced/`
- `artifacts/random_stratified_full_and_reduced/`
- top-level experiment reports such as `artifacts/meta_blend_experiment_report.json`

Important runtime boundary:

- REDUCED LightGBM and REDUCED blend remain offline-only.
- Random-stratified outputs remain alternate offline artifacts, not live API defaults.
- Offline experiment folders are not legal frontend runtime inputs.

## Frontend Integration Contract

Endpoints:

- `GET /health`
- `POST /score`

Request contract:

- REDUCED requests must contain only `application`
- FULL requests must contain `application`, `bureau_agg`, `previous_agg`, `installments_agg`, `pos_cash_agg`, and `credit_card_agg`

Response contract:

- `/health` returns `status`, `model_version`, `fairness_audit_passed`, `coverage_tiers_available`
- `/score` returns `probability_of_default`, `decision`, `escalate`, `top_5_explanations`, `model_version`, `calibrated`, `model_fairness_audit_passed`, `fairness_audit_version`, `coverage_tier`

Exact payload examples live in `Documentation/backend_api_contract.md`.

## Frontend Caveats

- `fairness_audit_passed` currently reflects the boolean stored in `artifacts/model_fairness_audit_passed.joblib`; in the checked-in runtime stack it is presently `false`.
- The health endpoint reports the FULL deployed version even though `/score` supports both FULL and REDUCED tiers.
- The API rejects partial FULL payloads, unknown top-level sections, malformed JSON, and forbidden `CODE_GENDER`.
- Runtime startup depends on the canonical top-level artifact names; moving frontend code to offline experiment folders will not work.
