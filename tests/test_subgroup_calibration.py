import numpy as np
import pandas as pd


def test_strategy_specs_cover_required_prototypes():
    from src.models.subgroup_calibration import _strategy_specs

    names = [spec.name for spec in _strategy_specs()]
    assert names == [
        'global_baseline',
        'region_aware',
        'targeted_worst_primary',
    ]


def test_grouped_calibration_strategy_routes_supported_groups_and_fallbacks():
    from src.models.subgroup_calibration import (
        _apply_grouped_calibration,
        _fit_grouped_calibration_strategy,
    )

    raw_pd = np.concatenate([
        np.linspace(0.05, 0.35, 60),
        np.linspace(0.10, 0.50, 45),
        np.linspace(0.20, 0.60, 15),
    ])
    y_true = np.concatenate([
        np.array([0] * 48 + [1] * 12),
        np.array([0] * 30 + [1] * 15),
        np.array([0] * 12 + [1] * 3),
    ])
    groups = pd.Series(['A'] * 60 + ['B'] * 45 + ['C'] * 15)

    strategy = _fit_grouped_calibration_strategy(
        y_true,
        raw_pd,
        groups,
        min_n=20,
        min_defaults=5,
        allowed_groups=('A', 'B'),
    )

    assert sorted(strategy['group_calibrators'].keys()) == ['A', 'B']
    fallback_reasons = {item['group']: item['reason'] for item in strategy['fallback_groups']}
    assert fallback_reasons['C'] == 'not_targeted'

    calibrated = _apply_grouped_calibration(strategy, raw_pd, groups)
    assert calibrated.shape == raw_pd.shape
    assert np.all((calibrated >= 0.0) & (calibrated <= 1.0))


def test_summarize_group_calibration_tracks_signed_gap_and_brier():
    from src.models.subgroup_calibration import _summarize_group_calibration

    frame = pd.DataFrame(
        {
            'FAIR_GROUP_PRIMARY': ['G1', 'G1', 'G2', 'G2'],
            'TARGET': [0, 1, 0, 1],
        }
    )
    calibrated_pd = np.array([0.1, 0.3, 0.7, 0.9])

    summary = _summarize_group_calibration(frame, 'FAIR_GROUP_PRIMARY', calibrated_pd)
    lookup = {row['group']: row for row in summary}

    assert lookup['G1']['n'] == 2
    assert np.isclose(lookup['G1']['signed_gap_obs_minus_pred'], 0.3)
    assert np.isclose(lookup['G2']['signed_gap_obs_minus_pred'], -0.3)
    assert lookup['G1']['brier_score'] > 0.0
    assert lookup['G2']['brier_score'] > 0.0


def test_select_target_groups_picks_worst_supported_primary_group(monkeypatch):
    from src.models import subgroup_calibration
    from src.models.subgroup_calibration import _select_target_groups, CalibrationStrategySpec

    monkeypatch.setattr(subgroup_calibration, 'GROUP_CALIBRATION_MIN_N', 20)
    monkeypatch.setattr(subgroup_calibration, 'GROUP_CALIBRATION_MIN_DEFAULTS', 5)

    groups = pd.Series(['REGION_1'] * 40 + ['REGION_2'] * 40 + ['REGION_3'] * 40)
    raw_pd = np.concatenate([
        np.full(40, 0.10),
        np.full(40, 0.20),
        np.full(40, 0.25),
    ])
    y_true = np.concatenate([
        np.array([0] * 36 + [1] * 4),
        np.array([0] * 30 + [1] * 10),
        np.array([0] * 20 + [1] * 20),
    ])

    allowed, details = _select_target_groups(
        CalibrationStrategySpec(
            name='targeted_worst_primary',
            description='test',
            group_column='FAIR_GROUP_PRIMARY',
            target_selection_policy='worst_primary_abs_gap',
        ),
        y_true=y_true,
        raw_pd=raw_pd,
        groups=groups,
    )

    assert allowed == ('REGION_3',)
    assert details['target_groups'] == ['REGION_3']
