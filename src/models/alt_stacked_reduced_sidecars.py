from __future__ import annotations

from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

SK_ID_CURR_COL = "SK_ID_CURR"
PARENT_SOURCE_FILE_COL = "PARENT_SOURCE_FILE"
SIDECAR_EXCLUDED_FEATURE_COLUMNS = frozenset({SK_ID_CURR_COL, PARENT_SOURCE_FILE_COL})

TELCO_UTILITY_REQUIRED_COLUMNS: tuple[str, ...] = (
    "SK_ID_CURR",
    "TELECOM_LATE_DAYS_MAX_6M",
    "UTILITY_DISCONNECT_FLAGS_12M",
    "PREPAID_TOP_UP_VOLATILITY",
    "TELECOM_ON_TIME_PAYMENT_RATE_6M",
    "UTILITY_LATE_PAYMENT_COUNT_12M",
    "AVG_MONTHLY_TOPUP_AMOUNT",
    "AVG_UTILITY_BILL_AMOUNT",
    "TELCO_PAYMENT_REGULARITY_SCORE",
    "UTILITY_STRESS_SCORE",
    "DIGITAL_SERVICE_STABILITY_SCORE",
)

WALLET_P2P_REQUIRED_COLUMNS: tuple[str, ...] = (
    "SK_ID_CURR",
    "P2P_MICRO_INFLOW_COUNT_30D",
    "WALLET_BALANCE_DEPLETION_RATE",
    "MERCHANT_PAYMENT_RATIO",
    "P2P_OUTFLOW_COUNT_30D",
    "DIGITAL_WALLET_CASHIN_COUNT_30D",
    "DIGITAL_WALLET_CASHOUT_COUNT_30D",
    "AVG_WALLET_TXN_AMOUNT",
    "WALLET_TXN_VOLATILITY",
    "WALLET_LIQUIDITY_STRESS_SCORE",
    "PEER_BORROWING_SIGNAL_SCORE",
    "FORMAL_MERCHANT_STABILITY_SCORE",
)

ECOMMERCE_SOCIAL_REQUIRED_COLUMNS: tuple[str, ...] = (
    "SK_ID_CURR",
    "CASH_ON_DELIVERY_RATIO",
    "SOCIAL_ACCOUNT_AGE_MONTHS",
    "HIGH_RISK_MERCHANT_TXNS",
    "ECOMMERCE_ORDER_COUNT_6M",
    "ECOMMERCE_RETURN_RATE",
    "AVG_BASKET_VALUE",
    "MERCHANT_CATEGORY_DIVERSITY",
    "SOCIAL_ENGAGEMENT_STABILITY_SCORE",
    "DIGITAL_TRUST_SCORE",
    "FRAUD_RISK_PROXY_SCORE",
)

SIDECAR_REQUIRED_COLUMNS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "telco_utility": TELCO_UTILITY_REQUIRED_COLUMNS,
        "wallet_p2p": WALLET_P2P_REQUIRED_COLUMNS,
        "ecommerce_social": ECOMMERCE_SOCIAL_REQUIRED_COLUMNS,
    }
)

__all__ = [
    "ECOMMERCE_SOCIAL_REQUIRED_COLUMNS",
    "PARENT_SOURCE_FILE_COL",
    "SIDECAR_REQUIRED_COLUMNS",
    "SK_ID_CURR_COL",
    "TELCO_UTILITY_REQUIRED_COLUMNS",
    "WALLET_P2P_REQUIRED_COLUMNS",
    "align_sidecar_to_ids",
    "extract_sidecar_feature_columns",
    "load_ecommerce_sidecar",
    "load_sidecar_csv",
    "load_telco_sidecar",
    "load_wallet_sidecar",
    "validate_ecommerce_sidecar",
    "validate_sidecar_frame",
    "validate_telco_sidecar",
    "validate_wallet_sidecar",
]


def _normalize_family_name(family_name: str) -> str:
    normalized = str(family_name).strip().lower()
    if normalized not in SIDECAR_REQUIRED_COLUMNS:
        supported = ", ".join(sorted(SIDECAR_REQUIRED_COLUMNS))
        raise ValueError(f"Unsupported sidecar family {family_name!r}. Expected one of: {supported}")
    return normalized


def _required_columns_for_family(family_name: str) -> tuple[str, ...]:
    return SIDECAR_REQUIRED_COLUMNS[_normalize_family_name(family_name)]


def _ensure_sk_id_curr_present(df: pd.DataFrame, family_name: str) -> None:
    if SK_ID_CURR_COL not in df.columns:
        raise ValueError(f"{family_name} sidecar is missing required column: {SK_ID_CURR_COL}")


def _assert_no_null_sk_ids(df: pd.DataFrame, family_name: str) -> None:
    if df[SK_ID_CURR_COL].isna().any():
        raise ValueError(f"{family_name} sidecar contains null SK_ID_CURR values")


def _assert_unique_sk_ids(df: pd.DataFrame, family_name: str) -> None:
    duplicate_mask = df[SK_ID_CURR_COL].duplicated(keep=False)
    if not duplicate_mask.any():
        return
    duplicate_ids = df.loc[duplicate_mask, SK_ID_CURR_COL].tolist()
    duplicate_preview = duplicate_ids[:10]
    raise ValueError(
        f"{family_name} sidecar contains duplicate SK_ID_CURR values: {duplicate_preview}"
    )


def _missing_required_columns(df: pd.DataFrame, required_columns: Sequence[str]) -> list[str]:
    return [column for column in required_columns if column not in df.columns]


def validate_sidecar_frame(
    sidecar_df: pd.DataFrame,
    family_name: str,
    required_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    normalized_family = _normalize_family_name(family_name)
    required = tuple(required_columns) if required_columns is not None else _required_columns_for_family(normalized_family)
    df = sidecar_df.copy()

    _ensure_sk_id_curr_present(df, normalized_family)
    missing_columns = _missing_required_columns(df, required)
    if missing_columns:
        raise ValueError(
            f"{normalized_family} sidecar is missing required columns: {missing_columns}"
        )

    _assert_no_null_sk_ids(df, normalized_family)
    _assert_unique_sk_ids(df, normalized_family)
    return df


def validate_telco_sidecar(sidecar_df: pd.DataFrame) -> pd.DataFrame:
    return validate_sidecar_frame(sidecar_df, "telco_utility", TELCO_UTILITY_REQUIRED_COLUMNS)


def validate_wallet_sidecar(sidecar_df: pd.DataFrame) -> pd.DataFrame:
    return validate_sidecar_frame(sidecar_df, "wallet_p2p", WALLET_P2P_REQUIRED_COLUMNS)


def validate_ecommerce_sidecar(sidecar_df: pd.DataFrame) -> pd.DataFrame:
    return validate_sidecar_frame(sidecar_df, "ecommerce_social", ECOMMERCE_SOCIAL_REQUIRED_COLUMNS)


def load_sidecar_csv(
    csv_path: str | Path,
    family_name: str,
    required_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    normalized_family = _normalize_family_name(family_name)
    path = Path(csv_path)
    try:
        sidecar_df = pd.read_csv(path)
    except Exception as exc:
        raise RuntimeError(f"Failed to load {normalized_family} sidecar CSV: {path}") from exc
    return validate_sidecar_frame(sidecar_df, normalized_family, required_columns)


def load_telco_sidecar(csv_path: str | Path) -> pd.DataFrame:
    return load_sidecar_csv(csv_path, "telco_utility", TELCO_UTILITY_REQUIRED_COLUMNS)


def load_wallet_sidecar(csv_path: str | Path) -> pd.DataFrame:
    return load_sidecar_csv(csv_path, "wallet_p2p", WALLET_P2P_REQUIRED_COLUMNS)


def load_ecommerce_sidecar(csv_path: str | Path) -> pd.DataFrame:
    return load_sidecar_csv(csv_path, "ecommerce_social", ECOMMERCE_SOCIAL_REQUIRED_COLUMNS)


def extract_sidecar_feature_columns(sidecar_df: pd.DataFrame) -> list[str]:
    return [
        column
        for column in sidecar_df.columns
        if column not in SIDECAR_EXCLUDED_FEATURE_COLUMNS
    ]


def align_sidecar_to_ids(
    master_ids_df: pd.DataFrame,
    sidecar_df: pd.DataFrame,
    family_name: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    normalized_family = _normalize_family_name(family_name)
    if SK_ID_CURR_COL not in master_ids_df.columns:
        raise ValueError(f"master_ids_df must contain {SK_ID_CURR_COL}")
    if master_ids_df[SK_ID_CURR_COL].isna().any():
        raise ValueError("master_ids_df contains null SK_ID_CURR values")

    validated_sidecar = validate_sidecar_frame(sidecar_df, normalized_family)
    master = master_ids_df.copy()
    master["__master_row_order"] = np.arange(len(master), dtype=np.int64)

    aligned = master.merge(
        validated_sidecar,
        on=SK_ID_CURR_COL,
        how="left",
        sort=False,
        validate="m:1",
        indicator="__sidecar_match",
    )
    aligned = aligned.sort_values("__master_row_order", kind="stable").reset_index(drop=True)

    matched_mask = aligned["__sidecar_match"].eq("both")
    unmatched_master_ids = aligned.loc[~matched_mask, SK_ID_CURR_COL].tolist()
    master_id_set = set(master[SK_ID_CURR_COL].tolist())
    sidecar_id_set = set(validated_sidecar[SK_ID_CURR_COL].tolist())
    unused_sidecar_ids = sorted(sidecar_id_set - master_id_set)
    feature_columns = extract_sidecar_feature_columns(validated_sidecar)

    metadata = {
        "family_name": normalized_family,
        "master_row_count": int(len(master)),
        "master_unique_id_count": int(master[SK_ID_CURR_COL].nunique(dropna=False)),
        "sidecar_row_count": int(len(validated_sidecar)),
        "sidecar_feature_columns": feature_columns,
        "matched_row_count": int(matched_mask.sum()),
        "unmatched_master_row_count": int((~matched_mask).sum()),
        "unmatched_master_ids": unmatched_master_ids,
        "unused_sidecar_row_count": int(len(unused_sidecar_ids)),
        "unused_sidecar_ids": unused_sidecar_ids,
        "coverage_rate": float(matched_mask.mean()) if len(master) else 0.0,
        "has_missing_coverage": bool((~matched_mask).any()),
    }

    aligned = aligned.drop(columns=["__master_row_order", "__sidecar_match"])
    return aligned, metadata
