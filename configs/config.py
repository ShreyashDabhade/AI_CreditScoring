"""
MasterMind — Shared Configuration
All project-wide constants. Every module imports from here.
"""

import os


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value

# ─── Reproducibility ─────────────────────────────────────
RANDOM_STATE: int = 42

# Debug / Sample Mode
DEBUG_MODE: bool = _env_flag("DEBUG_MODE", False)
DEBUG_SAMPLE_SIZE: int = _env_int("DEBUG_SAMPLE_SIZE", 50000)
DEBUG_CHUNK_SIZE: int = _env_int("DEBUG_CHUNK_SIZE", 100000)

# ─── Data Split Fractions ────────────────────────────────
TRAIN_FRAC: float = 0.60
VAL_MODEL_FRAC: float = 0.10
VAL_POLICY_FRAC: float = 0.10
TEST_FRAC: float = 0.20

# ─── Feature Engineering ─────────────────────────────────
RARE_CATEGORY_MIN_COUNT: int = 500
MISSING_RATE_THRESHOLD: float = 0.05

# ─── Decision Policy ─────────────────────────────────────
APPROVE_THRESHOLD: float = 0.15
DECLINE_THRESHOLD: float = 0.35

# ─── Fairness Audit ──────────────────────────────────────
FAIRNESS_MIN_N: int = 200
FAIRNESS_MIN_DEFAULTS: int = 20

# ─── Paths ───────────────────────────────────────────────
DATA_DIR: str = os.environ.get(
    "DATA_PROCESSED_DIR", "data/processed/")
ARTIFACT_DIR: str = os.environ.get(
    "ARTIFACT_DIR", "artifacts/")
RAW_DATA_DIR: str = os.environ.get(
    "DATA_RAW_DIR", "data/raw/")
DEBUG_RAW_DATA_DIR: str = os.environ.get(
    "DEBUG_RAW_DATA_DIR", "data/debug_raw/")

# ─── Versioning ──────────────────────────────────────────
FAIRNESS_AUDIT_VERSION: str = "proxy_audit_2026Q1_v1.0"
MODEL_VERSIONS: dict = {
    "full": "full_v2.1.0",
    "reduced": "reduced_v2.1.0",
}

# ─── API Section Requirements ────────────────────────────
FULL_REQUIRED_SECTIONS: set[str] = {
    "application", "bureau_agg", "previous_agg",
    "installments_agg", "pos_cash_agg", "credit_card_agg",
}


def resolve_runtime_raw_dir() -> str:
    # Always return the original raw data directory. Sampling is handled
    # in-memory by the pipeline when DEBUG_MODE is enabled; other modules
    # (feature engineering, builders) should read from the canonical
    # raw data location to avoid file-not-found errors for debug_raw.
    return RAW_DATA_DIR


# ─── Self-verification ──────────────────────────────────
if __name__ == "__main__":
    import importlib
    cfg = importlib.import_module("configs.config")
    assert cfg.RANDOM_STATE == 42
    assert cfg.APPROVE_THRESHOLD == 0.15
    assert cfg.DECLINE_THRESHOLD == 0.35
    assert cfg.FAIRNESS_AUDIT_VERSION == "proxy_audit_2026Q1_v1.0"
    assert "full" in cfg.MODEL_VERSIONS
    assert cfg.DEBUG_SAMPLE_SIZE > 0
    print("configs/config.py verified ✓")
