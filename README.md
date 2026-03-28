# MasterMind Credit Scoring

MasterMind is a credit-scoring repo organized as a strict offline pipeline plus a Flask scoring API.

- Modules 1-4 are offline:
  processed splits, frozen builders, trained/calibrated artifacts, and fairness/explainability outputs.
- Module 5 is the API:
  it eagerly validates and loads the persisted artifact stack once at startup.

This branch is frozen for handoff and presentation. The REDUCED modeling work has reached a truthful stopping point and no offline REDUCED candidate has been promoted into runtime defaults.

## Final Branch Status

- Active REDUCED runtime baseline remains XGBoost.
- REDUCED LightGBM exists only as an offline candidate.
- REDUCED weighted blend exists only as an offline candidate.
- Offline REDUCED blend experiments improved held-out ranking metrics directionally.
- Calibrated REDUCED candidates were too close to justify runtime promotion.
- Policy-threshold diagnostics showed threshold-dependent conclusions with no robust practical winner.
- Therefore this branch does not promote REDUCED LightGBM or REDUCED blend into live runtime paths.

## Runtime vs Offline Paths

Active runtime artifacts used by the API:

- `artifacts/reduced_model.joblib`
- `artifacts/reduced_calibrator.joblib`
- `artifacts/reduced_shap_explainer.joblib`
- `artifacts/full_model.joblib`
- `artifacts/full_calibrator.joblib`
- `artifacts/full_shap_explainer.joblib`
- validated FULL/REDUCED builders loaded through `src/builder_artifacts.py`

Offline-only REDUCED experiment directories:

- `artifacts/reduced_blend/`
- `artifacts/reduced_blend_calibrated_compare/`
- `artifacts/reduced_policy_threshold_diag/`

Those offline directories are evidence and diagnostics only. They are not API startup inputs and they do not replace runtime artifact names.

## Repo Workflow

The preserved workflow is:

1. Module 1 writes processed splits and `data/processed/processed_artifact_manifest.json`.
2. Module 2 fits and persists validated FULL/REDUCED frozen builders.
3. Module 3 persists trained/calibrated runtime artifacts plus `artifacts/reproducibility_report.json`.
4. Module 4 persists the fairness result artifact.
5. Module 5 starts only after eager validation of the persisted stack.

Real mode is strict and fail-fast by design. Missing lineage, wrong builders, or missing runtime artifacts should break startup.

## Installation

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`.

## Real-Mode API Prerequisites

Required before strict real-mode startup:

- `data/processed/processed_artifact_manifest.json`
- `artifacts/full_feature_builder.joblib`
- `artifacts/full_feature_builder.manifest.json`
- `artifacts/reduced_feature_builder.joblib`
- `artifacts/reduced_feature_builder.manifest.json`
- `artifacts/full_model.joblib`
- `artifacts/full_calibrator.joblib`
- `artifacts/full_shap_explainer.joblib`
- `artifacts/reduced_model.joblib`
- `artifacts/reduced_calibrator.joblib`
- `artifacts/reduced_shap_explainer.joblib`
- `artifacts/model_fairness_audit_passed.joblib`

Optional metadata:

- `artifacts/reproducibility_report.json`

Diagnostic-only processed artifacts such as `app_test_adv.pkl` are not live scoring inputs.

## Run The Demo API

Strict real mode:

```bash
export ARTIFACT_DIR=artifacts/
export DATA_PROCESSED_DIR=data/processed/
python -c "from src.api.app import create_app; app = create_app(); app.run(host='127.0.0.1', port=5000)"
```

Mock mode for local bring-up:

```bash
python -c "from src.api.app import create_app; app = create_app(mock_mode=True); app.run(host='127.0.0.1', port=5000)"
```

Then open:

- `/demo` for the interactive frontend
- `/health` for runtime metadata
- `/score` for strict JSON scoring

## API Contract

- `GET /health` returns runtime status, model version, fairness flag, and available tiers.
- `POST /score` supports:
  REDUCED application-only payloads
  and FULL payloads with all required aggregate sections.

The API does not load offline REDUCED experiment artifacts.

## Reproduce The Key Offline REDUCED Comparisons

Run these from the repo root after the real processed/runtime artifacts exist:

REDUCED blend search:

```bash
python -c "from src.models.reduced_blend import run_reduced_blend_experiment; run_reduced_blend_experiment()"
```

REDUCED calibrated comparison:

```bash
python -c "from src.models.reduced_blend_calibrated_compare import run_reduced_blend_calibrated_compare; run_reduced_blend_calibrated_compare()"
```

REDUCED policy-threshold diagnostic:

```bash
python -c "from src.models.reduced_policy_threshold_diag import run_reduced_policy_threshold_diag; run_reduced_policy_threshold_diag()"
```

REDUCED previous-vs-merged retraining comparison:

```bash
python -c "from src.models.reduced_training_regime_compare import run_reduced_training_regime_comparison; run_reduced_training_regime_comparison()"
```

FULL subgroup calibration experiments:

```bash
python -c "from src.models.subgroup_calibration import run_subgroup_calibration_experiments; run_subgroup_calibration_experiments()"
```

FULL fairness-aware retraining experiments:

```bash
python -c "from src.models.fairness_aware_training import run_fairness_aware_modeling_experiments; run_fairness_aware_modeling_experiments()"
```

The reports written by those commands remain offline-only and should not be treated as runtime defaults.

## Test Commands

Focused regression checks:

```bash
python -m pytest tests/test_api.py
python -m pytest tests/test_m3_blending.py
python -m pytest tests/test_reduced_blend_calibrated_compare.py
python -m pytest tests/test_reduced_policy_threshold_diag.py
```

## Key Documentation

- `Documentation/artifact_contract.md`
- `Documentation/final_branch_summary.md`
- `Documentation/offline_experiment_index.md`
- `howToRun.md`

## Truthful Final Position

- Active runtime REDUCED remains XGBoost.
- REDUCED LightGBM and REDUCED weighted blend remain offline candidates only.
- Blend is the strongest directional ranking candidate offline.
- Calibrated candidates are too close for a clear runtime winner.
- Policy usefulness is threshold-dependent.
- No REDUCED candidate is dominant enough for runtime promotion on this branch.
