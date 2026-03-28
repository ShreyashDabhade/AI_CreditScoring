import numpy as np
import pandas as pd


def test_strategy_specs_include_expected_region_interventions():
    from src.models.fairness_aware_training import _strategy_specs

    specs = _strategy_specs()
    names = [spec.name for spec in specs]
    assert names == ["region_balanced_ipw", "region_balanced_mild", "worst_region_default_boost"]


def test_region_balance_ipw_upweights_smaller_regions_more_aggressively():
    from src.models.fairness_aware_training import _region_balance_ipw_weights

    train_groups = pd.DataFrame(
        {
            "FAIR_GROUP_PRIMARY": [
                "REGION_1",
                "REGION_1",
                "REGION_2",
                "REGION_2",
                "REGION_2",
                "REGION_2",
                "REGION_3",
                "REGION_3",
            ]
        }
    )
    weights, details = _region_balance_ipw_weights(train_groups)

    np.testing.assert_allclose(weights.mean(), 1.0)
    assert details["region_counts"]["REGION_2"] == 4
    mean_by_region = (
        pd.DataFrame(
            {"region": train_groups["FAIR_GROUP_PRIMARY"], "weight": weights}
        )
        .groupby("region")["weight"]
        .mean()
    )
    assert mean_by_region["REGION_1"] > mean_by_region["REGION_2"]
    assert mean_by_region["REGION_1"] >= mean_by_region["REGION_3"]


def test_region_balance_weights_upweight_smaller_regions_and_normalize():
    from src.models.fairness_aware_training import _region_balance_weights

    train_groups = pd.DataFrame(
        {
            "FAIR_GROUP_PRIMARY": [
                "REGION_1",
                "REGION_1",
                "REGION_2",
                "REGION_2",
                "REGION_2",
                "REGION_2",
                "REGION_3",
                "REGION_3",
            ]
        }
    )
    weights, details = _region_balance_weights(train_groups)

    np.testing.assert_allclose(weights.mean(), 1.0)
    assert details["region_counts"]["REGION_2"] == 4
    mean_by_region = (
        pd.DataFrame(
            {"region": train_groups["FAIR_GROUP_PRIMARY"], "weight": weights}
        )
        .groupby("region")["weight"]
        .mean()
    )
    assert mean_by_region["REGION_1"] > mean_by_region["REGION_2"]
    assert mean_by_region["REGION_3"] > mean_by_region["REGION_2"]


def test_worst_region_default_boost_targets_only_worst_region_defaults_and_normalize():
    from src.models.fairness_aware_training import _worst_region_default_boost_weights

    train_groups = pd.DataFrame(
        {
            "FAIR_GROUP_PRIMARY": [
                "REGION_1",
                "REGION_2",
                "REGION_2",
                "REGION_2",
            ]
        }
    )
    y_train = np.array([0, 0, 1, 1], dtype=int)

    weights, details = _worst_region_default_boost_weights(train_groups, y_train)

    np.testing.assert_allclose(weights.mean(), 1.0)
    assert details["boosted_group"] == "REGION_2"
    assert details["boosted_rows"] == 2
    assert weights[2] > weights[0]
    assert weights[2] > weights[1]
    assert weights[0] == weights[1]
