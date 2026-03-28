import json
from pathlib import Path
from types import SimpleNamespace

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


def test_reduced_blend_formula_matches_weighted_average():
    from src.models.reduced_blend import blend_prob

    xgb = np.array([0.2, 0.8, 0.6])
    lgbm = np.array([0.4, 0.1, 0.9])

    blended = blend_prob(0.25, xgb, lgbm)

    np.testing.assert_allclose(blended, np.array([0.35, 0.275, 0.825]))


class _ReducedBuilderFixture:
    encoded_columns_ = ["xgb_prob", "lgbm_prob"]

    def transform(self, df):
        return df[self.encoded_columns_].astype(float).reset_index(drop=True)


class _ColumnProbModel:
    def __init__(self, column_index: int):
        self.column_index = column_index
        self.n_features_in_ = 2

    def predict_proba(self, X):
        arr = np.asarray(X, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        pd_col = np.clip(arr[:, self.column_index], 0.0, 1.0)
        return np.column_stack([1.0 - pd_col, pd_col])


def _make_reduced_blend_bundle():
    train = pd.DataFrame(
        {
            "xgb_prob": [0.12, 0.82, 0.22, 0.72, 0.35, 0.66],
            "lgbm_prob": [0.20, 0.74, 0.30, 0.69, 0.40, 0.62],
            "TARGET": [0, 1, 0, 1, 0, 1],
        }
    )
    val_model = pd.DataFrame(
        {
            "xgb_prob": [0.25, 0.58, 0.45, 0.81, 0.35, 0.77],
            "lgbm_prob": [0.10, 0.35, 0.62, 0.90, 0.30, 0.71],
            "TARGET": [0, 0, 1, 1, 0, 1],
        }
    )
    val_policy = pd.DataFrame(
        {
            "xgb_prob": [0.22, 0.61, 0.38, 0.84, 0.28, 0.75],
            "lgbm_prob": [0.14, 0.40, 0.64, 0.88, 0.25, 0.70],
            "TARGET": [0, 0, 1, 1, 0, 1],
        }
    )
    test = pd.DataFrame(
        {
            "xgb_prob": [0.20, 0.64, 0.36, 0.79, 0.33, 0.73],
            "lgbm_prob": [0.12, 0.39, 0.60, 0.87, 0.26, 0.69],
            "TARGET": [0, 0, 1, 1, 0, 1],
        }
    )
    return SimpleNamespace(
        mode="real",
        train=train,
        val_model=val_model,
        val_policy=val_policy,
        test=test,
        raw_dir="unused",
    )


def test_reduced_blend_experiment_persists_weight_search_results(tmp_path, monkeypatch):
    from src.models import reduced_blend

    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()

    monkeypatch.setattr(reduced_blend, "_has_real_training_inputs", lambda *args, **kwargs: True)
    monkeypatch.setattr(reduced_blend, "_load_real_bundle", lambda *args, **kwargs: _make_reduced_blend_bundle())
    monkeypatch.setattr(
        reduced_blend,
        "load_artifacts",
        lambda **kwargs: {
            "reduced_builder": _ReducedBuilderFixture(),
            "reduced_model": _ColumnProbModel(0),
        },
    )
    monkeypatch.setattr(
        reduced_blend,
        "_train_best_model_family",
        lambda *args, **kwargs: {
            "model": _ColumnProbModel(1),
            "selected_candidate": "stub_lgbm",
            "selected_params": {"n_estimators": 10},
            "val_model_roc_auc": 0.0,
            "val_model_raw_pd": _make_reduced_blend_bundle().val_model["lgbm_prob"].to_numpy(dtype=float),
        },
    )
    monkeypatch.setattr(reduced_blend, "_processed_manifest_fingerprint", lambda *args, **kwargs: "processed-xyz")

    report = reduced_blend.run_reduced_blend_experiment(
        artifact_dir=str(artifact_dir),
        processed_dir=str(tmp_path / "processed"),
        raw_dir=str(tmp_path / "raw"),
    )

    report_path = artifact_dir / "reduced_blend" / "reduced_blend_report.json"
    summary_path = artifact_dir / "reduced_blend" / "reduced_blend_summary.md"
    assert report_path.exists()
    assert summary_path.exists()

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    weights = payload["blend_search"]["weights"]
    assert len(weights) > 11
    assert payload["blend_search"]["coarse_grid"] == [round(step / 10.0, 1) for step in range(11)]
    assert payload["blend_search"]["selected_weight_xgb"] == report["blend_search"]["selected_weight_xgb"]
    assert any(row["selection_stage"] == "refine" for row in weights)
    assert all("test_metrics" in row for row in weights)


def test_reduced_blend_experiment_keeps_runtime_artifacts_isolated(tmp_path, monkeypatch):
    from src.models import reduced_blend

    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    reduced_model_path = artifact_dir / "reduced_model.joblib"
    reduced_calibrator_path = artifact_dir / "reduced_calibrator.joblib"
    reduced_model_path.write_text("runtime-model-sentinel", encoding="utf-8")
    reduced_calibrator_path.write_text("runtime-calibrator-sentinel", encoding="utf-8")

    monkeypatch.setattr(reduced_blend, "_has_real_training_inputs", lambda *args, **kwargs: True)
    monkeypatch.setattr(reduced_blend, "_load_real_bundle", lambda *args, **kwargs: _make_reduced_blend_bundle())
    monkeypatch.setattr(
        reduced_blend,
        "load_artifacts",
        lambda **kwargs: {
            "reduced_builder": _ReducedBuilderFixture(),
            "reduced_model": _ColumnProbModel(0),
        },
    )
    monkeypatch.setattr(
        reduced_blend,
        "_train_best_model_family",
        lambda *args, **kwargs: {
            "model": _ColumnProbModel(1),
            "selected_candidate": "stub_lgbm",
            "selected_params": {"n_estimators": 10},
            "val_model_roc_auc": 0.0,
            "val_model_raw_pd": _make_reduced_blend_bundle().val_model["lgbm_prob"].to_numpy(dtype=float),
        },
    )
    monkeypatch.setattr(reduced_blend, "_processed_manifest_fingerprint", lambda *args, **kwargs: "processed-xyz")

    report = reduced_blend.run_reduced_blend_experiment(
        artifact_dir=str(artifact_dir),
        processed_dir=str(tmp_path / "processed"),
        raw_dir=str(tmp_path / "raw"),
    )

    isolated_dir = artifact_dir / "reduced_blend"
    assert isolated_dir.exists()
    assert reduced_model_path.read_text(encoding="utf-8") == "runtime-model-sentinel"
    assert reduced_calibrator_path.read_text(encoding="utf-8") == "runtime-calibrator-sentinel"
    assert (isolated_dir / "reduced_blend_selected_model.joblib").exists()
    assert (isolated_dir / "reduced_blend_selected_calibrator.joblib").exists()
    assert report["runtime_unchanged"] is True
    assert report["default_runtime_artifacts_mutated"] is False


def test_build_training_regime_bundle_can_merge_train_and_val_policy():
    from src.models.train import DatasetBundle, _build_training_regime_bundle

    bundle = DatasetBundle(
        mode="real",
        train=pd.DataFrame({"feature": [1, 2], "TARGET": [0, 1]}),
        val_model=pd.DataFrame({"feature": [3], "TARGET": [0]}),
        val_policy=pd.DataFrame({"feature": [4, 5], "TARGET": [1, 0]}),
        test=pd.DataFrame({"feature": [6], "TARGET": [1]}),
        raw_dir="data/raw",
        uses_flattened_full_input=True,
    )

    merged_bundle, details = _build_training_regime_bundle(bundle, "train_plus_val_policy")

    assert len(merged_bundle.train) == 4
    assert merged_bundle.train["feature"].tolist() == [1, 2, 4, 5]
    assert merged_bundle.val_model["feature"].tolist() == [3]
    assert merged_bundle.val_policy["feature"].tolist() == [3]
    assert details["train_split_sources"] == ["train", "val_policy"]
    assert details["calibration_split"] == "val_model"
