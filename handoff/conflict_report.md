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

## CONFLICT-02: Historical Module 4 Dependency On Module 3 Artifacts
**Type**: historical note
**Producer**: Module 3 - `src/models/train.py`
**Consumer**: Module 4 - `src/fairness_audit.py`
**Description**: Earlier integration notes recorded that the real-data Module 4 orchestration path depended on `load_artifacts()` and `decision_from_pd()` from Module 3 before those symbols were fully wired.
**Resolution**: This is now superseded. `src/models/train.py` exports both symbols, the real-data fairness audit path executes end to end, and current verification covers the Module 3 to Module 4 runtime boundary.
**Stage to fix**: Superseded
**Status**: RESOLVED
**Reason**: Keep this entry only as historical context so the repo no longer suggests that the current real-data Module 4 path is still blocked.

## No-conflict confirmations
- M2 exports `pool_rare_categories()`, and Module 4 consumes it directly inside `derive_fairness_groups()` without signature mismatch.
- Module 2's current builder-based contract (`FrozenFeatureBuilder`, `fit_full_builder()`, `build_full()`) matches Module 4's integration notes and real-data orchestration path.
- Module 2 drops fairness raw columns from the model matrix, and Module 4 correctly builds its fairness audit frame from raw split columns instead of `build_full()` output.
