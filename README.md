# MasterMind Credit Scoring

MasterMind is a credit-scoring backend with a strict offline pipeline and a Flask scoring API.

This branch is frozen for backend/frontend merge readiness. The goal is runtime stability, not further model iteration.

## Frozen Runtime Defaults

- Default processed runtime path: `data/processed/`
- Default artifact runtime path: `artifacts/`
- Supported split modes: `proxy_time`, `random_stratified`
- Current default split mode: `proxy_time`

The real API loads only the canonical runtime artifact names from the default runtime paths unless `artifact_dir` or `processed_dir` is explicitly overridden.

## Active Runtime Artifact Stack

Required processed artifact:

- `data/processed/processed_artifact_manifest.json`

Required builder artifacts:

- `artifacts/full_feature_builder.joblib`
- `artifacts/full_feature_builder.manifest.json`
- `artifacts/reduced_feature_builder.joblib`
- `artifacts/reduced_feature_builder.manifest.json`

Required scoring artifacts:

- `artifacts/full_model.joblib`
- `artifacts/full_calibrator.joblib`
- `artifacts/full_shap_explainer.joblib`
- `artifacts/reduced_model.joblib`
- `artifacts/reduced_calibrator.joblib`
- `artifacts/reduced_shap_explainer.joblib`
- `artifacts/model_fairness_audit_passed.joblib`

Optional runtime metadata:

- `artifacts/reproducibility_report.json`

Current deployed versions in this branch:

- FULL runtime version: `full_weighted_blend_v2.2.0`
- REDUCED runtime version: `reduced_v2.1.0`

Important:

- FULL is already deployed in this branch through the canonical runtime files `artifacts/full_model.joblib`, `artifacts/full_calibrator.joblib`, and `artifacts/full_shap_explainer.joblib`.
- REDUCED runtime remains the canonical single-model path under `artifacts/reduced_model.joblib`, `artifacts/reduced_calibrator.joblib`, and `artifacts/reduced_shap_explainer.joblib`.
- REDUCED LightGBM, REDUCED blend, competition-mode artifacts, and other research outputs are offline-only and are not API inputs.

## API Contract

- `GET /health` returns:
  `status`, `model_version`, `fairness_audit_passed`, `coverage_tiers_available`
- `POST /score` accepts:
  REDUCED payloads with only `application`
- `POST /score` also accepts:
  FULL payloads with `application`, `bureau_agg`, `previous_agg`, `installments_agg`, `pos_cash_agg`, and `credit_card_agg`

The API is strict by design:

- partial FULL payloads are rejected
- unknown top-level sections are rejected
- malformed JSON returns `400`
- contract violations return `422`

Exact request and response examples are documented in [Documentation/backend_api_contract.md](Documentation/backend_api_contract.md).

## Runtime Vs Offline

Canonical runtime inputs live only at the top level of:

- `data/processed/`
- `artifacts/`

Offline-only experiment outputs preserved in this branch include:

- `artifacts/reduced_thin_blend/`
- `artifacts/reduced_thin_lgbm/`
- `artifacts/reduced_thin_diag/`
- `artifacts/reduced_thin_opt/`
- `artifacts/alt_stacked_reduced/`
- `artifacts/random_stratified_reduced/`
- `artifacts/random_stratified_full_and_reduced/`
- top-level offline reports such as `artifacts/meta_blend_experiment_report.json`

Those paths are evidence only. They must not be treated as live runtime defaults.

## Running The Real API

```bash
python -c "from src.api.app import create_app; app = create_app(mock_mode=False, strict_artifacts=True); app.run(host='127.0.0.1', port=5000)"
```

Useful endpoints:

- `/demo`
- `/health`
- `/score`

Mock mode remains available for local UI bring-up:

```bash
python -c "from src.api.app import create_app; app = create_app(mock_mode=True); app.run(host='127.0.0.1', port=5000)"
```

## Focused Regression Checks

```bash
python -m pytest tests/test_api.py
python -m pytest tests/test_backend_freeze_contract.py
python -m pytest tests/test_module1_split_regime.py
```

## Key Documentation

- `Documentation/backend_api_contract.md`
- `Documentation/backend_freeze_summary.md`
- `Documentation/artifact_contract.md`
- `Documentation/offline_experiment_index.md`
- `howToRun.md`
