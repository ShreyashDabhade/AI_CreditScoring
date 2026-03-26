import json
from pathlib import Path

import numpy as np
import pandas as pd


def test_weighted_average_blend_can_choose_interior_weight():
    from src.models.train import _select_weighted_average_blend

    y_true = np.array([0, 0, 1, 1, 0, 1])
    pred_xgb = np.array([0.1, 0.45, 0.55, 0.9, 0.35, 0.8])
    pred_lgbm = np.array([0.05, 0.2, 0.7, 0.85, 0.4, 0.6])

    result = _select_weighted_average_blend(y_true, pred_xgb, pred_lgbm, weights=np.array([0.0, 0.25, 0.5, 0.75, 1.0]))

    assert 0.0 <= result["weight_xgb"] <= 1.0
    assert 0.0 <= result["weight_lgbm"] <= 1.0
    np.testing.assert_allclose(result["weight_xgb"] + result["weight_lgbm"], 1.0)
    assert result["val_model_roc_auc"] >= 0.5


def test_reproducibility_report_annotation_updates_blend_status(tmp_path):
    from src.models.train import _annotate_reproducibility_report_with_blend_evaluation

    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_path = artifact_dir / "reproducibility_report.json"
    report_path.write_text(json.dumps({"mode": "real", "blend_evaluation": {"evaluated": False, "report_path": None, "best_candidate": None, "deployed": False}}), encoding="utf-8")

    _annotate_reproducibility_report_with_blend_evaluation(
        str(artifact_dir),
        evaluated=True,
        blend_report_path=str(artifact_dir / "blend_experiment_report.json"),
        best_candidate="weighted_blend_full",
    )

    updated = json.loads(report_path.read_text(encoding="utf-8"))
    assert updated["blend_evaluation"]["evaluated"] is True
    assert updated["blend_evaluation"]["best_candidate"] == "weighted_blend_full"


class _StaticProbModel:
    def __init__(self, probs):
        self._probs = np.asarray(probs, dtype=float)

    def predict_proba(self, X):
        arr = np.asarray(X, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        probs = np.resize(self._probs, arr.shape[0])
        return np.column_stack([1.0 - probs, probs])


class _StaticExplainer:
    def __init__(self, values):
        self._values = np.asarray(values, dtype=float)

    def __call__(self, X):
        arr = np.asarray(X, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if self._values.ndim == 1:
            return np.tile(self._values, (arr.shape[0], 1))
        return self._values


def test_weighted_blend_runtime_wrappers_combine_components():
    from src.models.runtime_support import WeightedBlendModel, WeightedBlendShapExplainer

    features = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], columns=["f1", "f2"])
    model = WeightedBlendModel(
        _StaticProbModel([0.2, 0.8]),
        _StaticProbModel([0.4, 0.6]),
        weight_xgboost=0.4,
        weight_lightgbm=0.6,
        feature_count=2,
    )
    probs = model.predict_proba(features)
    np.testing.assert_allclose(probs[:, 1], np.array([0.32, 0.68]))
    assert model.n_features_in_ == 2

    explainer = WeightedBlendShapExplainer(
        _StaticExplainer([[1.0, 3.0], [1.0, 3.0]]),
        _StaticExplainer([[2.0, 1.0], [2.0, 1.0]]),
        weight_xgboost=0.4,
        weight_lightgbm=0.6,
    )
    shap_values = explainer(features.to_numpy(dtype=float))
    np.testing.assert_allclose(shap_values, np.array([[1.6, 1.8], [1.6, 1.8]]))


def test_full_runtime_candidate_persists_weighted_blend_metadata(tmp_path):
    from src.models.train import _train_full_runtime_candidate

    rng = np.random.default_rng(42)
    frame = pd.DataFrame(rng.normal(size=(120, 6)), columns=[f"f_{i}" for i in range(6)])
    score = frame["f_0"] + 0.8 * frame["f_1"] - 0.4 * frame["f_2"]
    target = (score > np.median(score)).astype(int).to_numpy()

    result = _train_full_runtime_candidate(
        frame.iloc[:60].reset_index(drop=True),
        target[:60],
        frame.iloc[60:80].reset_index(drop=True),
        target[60:80],
        frame.iloc[80:100].reset_index(drop=True),
        target[80:100],
        frame.iloc[100:120].reset_index(drop=True),
        target[100:120],
        str(tmp_path),
        full_feature_view="FULL_COMPLETE",
        processed_manifest_fingerprint="processed-123",
    )

    metadata_path = tmp_path / 'full_weighted_blend_metadata.json'
    assert metadata_path.exists()
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    assert metadata['processed_manifest_fingerprint'] == 'processed-123'
    assert metadata['selected_runtime_candidate'] in {'xgboost_full', 'weighted_blend_full'}
    assert (tmp_path / 'full_model.joblib').exists()
    assert (tmp_path / 'full_xgboost_model.joblib').exists()
    assert (tmp_path / 'full_weighted_blend_model.joblib').exists()
    assert result['selected_candidate'] in {'xgboost_full', 'weighted_blend_full'}
