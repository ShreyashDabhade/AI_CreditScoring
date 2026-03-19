## Module 5 - Flask API: Integration Notes

### What this module does
Module 5 exposes the scoring API as the terminal consumer of the repaired offline workflow.

- `create_app()` eagerly validates and loads the artifact stack once in real mode.
- `POST /score` validates payloads, routes FULL vs REDUCED coverage, transforms through loaded builders, scores, calibrates, decides, explains, and returns the exact 9-field response.
- `GET /health` reports runtime availability and deployed model metadata.

### Hard dependencies
- `data/processed/processed_artifact_manifest.json`
- strict builder loading through `src/builder_artifacts.py`
- persisted FULL and REDUCED builders validated against the current processed lineage
- trained model artifacts
- calibrator artifacts
- SHAP explainer artifacts
- `artifacts/model_fairness_audit_passed.joblib`

### Important workflow rules
- Module 5 must not rebuild or refit builders in normal mode.
- Module 5 must not read raw child CSVs at request time.
- Module 5 must not depend on `app_test_adv.pkl` for live scoring. That file is diagnostic-only.
- Real mode is eager and fail-fast. Missing lineage, missing artifacts, or incompatible artifacts must stop startup immediately.
- Mock mode is for tests and local route bring-up only. It must remain temp-scoped by default and must not write to shared repo artifact paths.

### Runtime boundary
Module 5 stores one immutable runtime object on the Flask app after startup validation. Requests consume only that loaded runtime object and must not reload artifacts or mutate runtime state per request.

### Public API contract
- `create_app(artifact_dir=None, processed_dir=None, mock_mode=False, strict_artifacts=True)`
- `validate_payload(payload)`
- `determine_coverage_tier(payload)`
- `build_input_df(payload, tier)`
- `score_request(payload, runtime, mock_mode=False)`

### Notes for other modules
- Module 5 expects Module 1 processed lineage to be authoritative.
- Module 5 expects Module 2 builder lineage validation to be authoritative through `src/builder_artifacts.py`.
- Module 5 expects Module 3 persisted model, calibrator, and explainer artifacts to exist before real-mode startup.
- Module 5 expects Module 4 fairness output to already exist and only reads the persisted fairness result.

### Completion check
Module 5 should be considered incomplete if `src/api/app.py` or `tests/test_api.py` is still a stub in the working tree.
