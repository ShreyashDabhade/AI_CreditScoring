import pandas as pd
import pytest

from src.data_pipeline import ordered_split_60_10_10_20, proxy_recency_sort
from src.runtime_verification import validate_processed_splits


def _make_split_frame(sk_id: int, days_id_publish: float, days_registration: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "SK_ID_CURR": [sk_id],
            "TARGET": [0],
            "DAYS_ID_PUBLISH": [days_id_publish],
            "DAYS_REGISTRATION": [days_registration],
            "DAYS_EMPLOYED_ANOM": [0],
        }
    )


def test_proxy_recency_sort_orders_oldest_rows_first_with_stable_tie_breaks():
    df = pd.DataFrame(
        {
            "SK_ID_CURR": [30, 10, 20, 40],
            "DAYS_ID_PUBLISH": [-100.0, -400.0, -400.0, -200.0],
            "DAYS_REGISTRATION": [-150.0, -450.0, -300.0, -250.0],
        }
    )

    result = proxy_recency_sort(df)

    assert result["SK_ID_CURR"].tolist() == [10, 20, 40, 30]


def test_ordered_split_and_verifier_enforce_oldest_to_newest_regime():
    ordered = pd.concat(
        [
            _make_split_frame(100001, -5000.0, -6000.0),
            _make_split_frame(100002, -4000.0, -5000.0),
            _make_split_frame(100003, -3000.0, -4000.0),
            _make_split_frame(100004, -2000.0, -3000.0),
            _make_split_frame(100005, -1000.0, -2000.0),
            _make_split_frame(100006, -500.0, -1000.0),
            _make_split_frame(100007, -400.0, -900.0),
            _make_split_frame(100008, -300.0, -800.0),
            _make_split_frame(100009, -200.0, -700.0),
            _make_split_frame(100010, -100.0, -600.0),
        ],
        ignore_index=True,
    )

    train, val_model, val_policy, test = ordered_split_60_10_10_20(ordered)
    verification = validate_processed_splits(
        {
            "train": train,
            "val_model": val_model,
            "val_policy": val_policy,
            "test": test,
        }
    )

    assert verification["split_summary"]["train"]["mean_abs_days_id_publish"] >= verification["split_summary"]["val_model"]["mean_abs_days_id_publish"]
    assert verification["split_summary"]["val_model"]["mean_abs_days_id_publish"] >= verification["split_summary"]["val_policy"]["mean_abs_days_id_publish"]
    assert verification["split_summary"]["val_policy"]["mean_abs_days_id_publish"] >= verification["split_summary"]["test"]["mean_abs_days_id_publish"]

    wrong_order = {
        "train": test,
        "val_model": val_policy,
        "val_policy": val_model,
        "test": train,
    }
    with pytest.raises(ValueError, match="Proxy-time split ordering violated"):
        validate_processed_splits(wrong_order)
