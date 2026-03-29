# Offline Experiment Index

These outputs are preserved for offline analysis only. They are not runtime API inputs.

## Alternate REDUCED Runtime Candidates

- `artifacts/reduced_thin_blend/`
  contains `reduced_thin_blend_report.json`, `reduced_thin_blend_summary.md`, and selected blend sidecar artifacts
- `artifacts/reduced_thin_lgbm/`
  contains REDUCED LightGBM candidate runs and summaries
- `artifacts/reduced_thin_diag/`
  contains threshold and policy diagnostic summaries
- `artifacts/reduced_thin_opt/`
  contains alternative REDUCED thin-feature optimization artifacts

## Alternate Data Or Stacking Experiments

- `artifacts/alt_stacked_reduced/`
  contains stacked-sidecar REDUCED experiments and reports
- `data/processed/alt_stacked_reduced/`
  contains processed data prepared for that offline stack
- `data/processed/thin_file_alt/`
  contains alternative processed inputs for thin-file experiments

## Alternate Split-Mode Outputs

- `artifacts/random_stratified_reduced/`
  competition-mode REDUCED outputs for the `random_stratified` split regime
- `artifacts/random_stratified_full_and_reduced/`
  alternate FULL and REDUCED artifact stack built against `random_stratified`
- `data/processed_random_stratified/`
  processed manifest and splits for the alternate `random_stratified` regime

## Top-Level Offline Reports

- `artifacts/meta_blend_experiment_report.json`
- `artifacts/catboost_experiment_report.json`
- `artifacts/cv_blend_weight_experiment_report.json`
- `artifacts/feature_pruning_experiment_report.json`
- `artifacts/policy_threshold_sensitivity_report.json`
- `artifacts/regularization_experiment_report.json`
- `artifacts/subgroup_calibration_experiment_report.json`
- `artifacts/fairness_aware_modeling_experiment_report.json`

## Guardrail

- Canonical runtime inputs remain only `data/processed/` plus top-level `artifacts/` runtime files.
- Offline folders must not replace `artifacts/full_model.joblib`, `artifacts/reduced_model.joblib`, or the canonical builder/calibrator/explainer files used by the API.
