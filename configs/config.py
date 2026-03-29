"""
MasterMind â€” Shared Configuration
All project-wide constants. Every module imports from here.
"""

import os

# â”€â”€â”€ Reproducibility â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
RANDOM_STATE: int = 42

# Split regime
DEFAULT_SPLIT_MODE: str = "proxy_time"
SUPPORTED_SPLIT_MODES: tuple[str, ...] = ("proxy_time", "random_stratified")

# â”€â”€â”€ Data Split Fractions â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
TRAIN_FRAC: float = 0.60
VAL_MODEL_FRAC: float = 0.10
VAL_POLICY_FRAC: float = 0.10
TEST_FRAC: float = 0.20

# â”€â”€â”€ Feature Engineering â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
RARE_CATEGORY_MIN_COUNT: int = 500
MISSING_RATE_THRESHOLD: float = 0.05

# â”€â”€â”€ Decision Policy â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
APPROVE_THRESHOLD: float = 0.15
DECLINE_THRESHOLD: float = 0.35

# â”€â”€â”€ Fairness Audit â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
FAIRNESS_MIN_N: int = 200
FAIRNESS_MIN_DEFAULTS: int = 20

# â”€â”€â”€ Paths â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
DATA_DIR: str = os.environ.get(
    "DATA_PROCESSED_DIR", "data/processed/")
ARTIFACT_DIR: str = os.environ.get(
    "ARTIFACT_DIR", "artifacts/")

# â”€â”€â”€ Versioning â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
FAIRNESS_AUDIT_VERSION: str = "proxy_audit_2026Q1_v1.1"
MODEL_VERSIONS: dict = {
    "full": "full_v2.1.0",
    "reduced": "reduced_v2.1.0",
}

# â”€â”€â”€ API Section Requirements â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
FULL_REQUIRED_SECTIONS: set[str] = {
    "application", "bureau_agg", "previous_agg",
    "installments_agg", "pos_cash_agg", "credit_card_agg",
}


# â”€â”€â”€ Self-verification â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if __name__ == "__main__":
    import importlib
    cfg = importlib.import_module("configs.config")
    assert cfg.RANDOM_STATE == 42
    assert cfg.APPROVE_THRESHOLD == 0.15
    assert cfg.DECLINE_THRESHOLD == 0.35
    assert cfg.FAIRNESS_AUDIT_VERSION == "proxy_audit_2026Q1_v1.1"
    assert "full" in cfg.MODEL_VERSIONS
    print("configs/config.py verified âœ“")

