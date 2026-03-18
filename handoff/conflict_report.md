# MasterMind Integration Conflict Report

## Summary
Total conflicts found: 2
Conflicts requiring code changes: 1
Conflicts resolvable by wiring only: 1

## CONFLICT-01: Full Builder Loading Is Implicit and Unvalidated
**Type**: integration gap
**Producer**: Module 2 — `artifacts/full_feature_builder.joblib`
**Consumer**: Module 4 — `src/fairness_audit.py`
**Description**: Module 4 depended on Module 2's persisted `FrozenFeatureBuilder`, but the real-data path loaded the artifact inline via a hardcoded path and did not validate that the loaded object matched the Module 2 builder contract.
**Resolution**: Add a private boundary shim in `src/fairness_audit.py` that loads or fits the Module 2 full builder, validates the loaded object type, and optionally saves it to a caller-provided path for isolated integration tests.
**Stage to fix**: Stage 3
**Resolution applied**: `src/fairness_audit.py` — added `_load_or_fit_full_builder(...)` and switched the real-data orchestration path to use it.
**Status**: RESOLVED

## CONFLICT-02: Real-Data Module 4 Path Still Depends on Module 3 Artifacts
**Type**: missing implementation
**Producer**: Module 3 — `src/models/train.py`
**Consumer**: Module 4 — `src/fairness_audit.py`
**Description**: The real-data Module 4 orchestration path still requires `load_artifacts()` and `decision_from_pd()` from Module 3. Current `main` does not yet provide those symbols, so end-to-end real-data Module 4 execution remains blocked outside the M2↔M4 boundary.
**Resolution**: Leave the M2↔M4 integration intact, verify it with isolated synthetic tests, and defer the real-data runtime hookup until Module 3 lands.
**Stage to fix**: Stage 3
**Status**: FLAGGED — requires manual review
**Reason**: Resolving this now would require implementing Module 3 internals, which is outside the requested M2↔M4 integration slice.

## No-conflict confirmations
- M2 exports `pool_rare_categories()`, and Module 4 consumes it directly inside `derive_fairness_groups()` without signature mismatch.
- Module 2's current builder-based contract (`FrozenFeatureBuilder`, `fit_full_builder()`, `build_full()`) matches Module 4's integration notes and real-data orchestration path.
- Module 2 drops fairness raw columns from the model matrix, and Module 4 correctly builds its fairness audit frame from raw split columns instead of `build_full()` output.
