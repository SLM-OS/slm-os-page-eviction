"""Unit tests for training pipeline helpers.

Tests the pure-Python pieces of the training pipeline (grid search scaffolding,
dataset manipulation for reduced feature sets). Model training itself is too
expensive for unit tests — those are exercised by integration scripts
(scripts/train_all.py, scripts/feature_reduction.py).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.training.dataset import EvictionDataset


def _make_mock_dataset(
    num_rows: int = 100,
    num_features: int = 27,
    num_evictions: int = 10,
    scenarios: list[str] | None = None,
) -> EvictionDataset:
    """Construct a small synthetic EvictionDataset for testing."""
    rng = np.random.default_rng(42)
    features = rng.uniform(0, 1, size=(num_rows, num_features)).astype(np.float32)

    # Exactly one optimal per eviction group
    is_optimal = np.zeros(num_rows, dtype=np.int8)
    eviction_ids = np.repeat(
        np.arange(num_evictions), num_rows // num_evictions
    )[:num_rows]
    for eid in range(num_evictions):
        mask = eviction_ids == eid
        idx = np.where(mask)[0]
        if len(idx) > 0:
            is_optimal[idx[0]] = 1

    reuse_dist = rng.uniform(0, 1, size=num_rows).astype(np.float32)
    if scenarios is None:
        scenarios = ["scenario_a", "scenario_b"]
    scenario_arr = np.array(
        [scenarios[i % len(scenarios)] for i in range(num_rows)]
    )

    feature_names = [f"f{i}" for i in range(num_features)]

    return EvictionDataset(
        features=features,
        is_optimal=is_optimal,
        reuse_distance=reuse_dist,
        eviction_ids=eviction_ids,
        scenarios=scenario_arr,
        feature_names=feature_names,
    )


class TestXGBoostGridSearch:
    """Tests for the grid_search_xgb helper added during Phase 3 tuning."""

    def test_grid_search_returns_sorted_results(self):
        xgboost = pytest.importorskip("xgboost")
        from src.training.train_xgb import grid_search_xgb

        train = _make_mock_dataset(num_rows=200, num_evictions=20)
        val = _make_mock_dataset(num_rows=50, num_evictions=5)

        # Tiny grid — just verify structure
        grid = {"max_depth": [3, 4], "learning_rate": [0.1]}
        results = grid_search_xgb(train, val, param_grid=grid)

        assert len(results) == 2
        # Sorted by val_auc descending
        assert results[0]["val_auc"] >= results[1]["val_auc"]
        # Each entry has the params and metrics
        for entry in results:
            assert "max_depth" in entry
            assert "learning_rate" in entry
            assert "val_auc" in entry
            assert "best_iteration" in entry

    def test_grid_search_default_grid(self):
        """Default grid has 4x3x3 = 36 configurations but we skip this full-run test."""
        from src.training.train_xgb import grid_search_xgb
        import inspect

        sig = inspect.signature(grid_search_xgb)
        assert "param_grid" in sig.parameters
        assert sig.parameters["param_grid"].default is None


class TestReducedFeatureSelection:
    """Tests for the feature selection helper in scripts/feature_reduction.py."""

    def test_select_features_preserves_labels(self):
        from scripts.feature_reduction import select_features

        data = _make_mock_dataset(num_rows=100, num_features=10)
        selected = ["f0", "f2", "f4"]
        sub = select_features(data, selected)

        assert sub.features.shape == (100, 3)
        assert sub.feature_names == selected
        # Labels preserved
        assert np.array_equal(sub.is_optimal, data.is_optimal)
        assert np.array_equal(sub.eviction_ids, data.eviction_ids)

    def test_select_features_correct_columns(self):
        from scripts.feature_reduction import select_features

        data = _make_mock_dataset(num_rows=10, num_features=5)
        selected = ["f2", "f0"]
        sub = select_features(data, selected)

        # First column of sub should be data.features[:, 2]
        np.testing.assert_array_equal(sub.features[:, 0], data.features[:, 2])
        np.testing.assert_array_equal(sub.features[:, 1], data.features[:, 0])


class TestDAggerOnPolicyData:
    """Tests for DAgger on-policy data collection added in Phase 4."""

    def test_dagger_imports_resolve(self):
        """Regression: missing MemoryState import caused NameError on run."""
        # Just importing the module should succeed; broken imports would fail
        from src.training.dagger import dagger_finetune, _collect_on_policy_data
        assert callable(dagger_finetune)
        assert callable(_collect_on_policy_data)

    def test_merge_datasets_respects_mix_ratio(self):
        from src.training.dagger import _merge_datasets

        original = _make_mock_dataset(num_rows=100, num_evictions=10)
        new = _make_mock_dataset(num_rows=50, num_evictions=5)

        merged = _merge_datasets(original, new, mix_ratio=0.5)
        # Original 100 + 50% of 50 = 125 rows
        assert merged.num_rows == 125

    def test_merge_datasets_mix_ratio_zero(self):
        from src.training.dagger import _merge_datasets

        original = _make_mock_dataset(num_rows=100)
        new = _make_mock_dataset(num_rows=50)
        merged = _merge_datasets(original, new, mix_ratio=0.0)
        assert merged.num_rows == original.num_rows


class TestMLPFeatureConfigExposure:
    """MLPPolicy should expose feature_config for DAgger use."""

    def test_feature_config_property_exists(self):
        """Regression: DAgger needs access to feature_config to build extractor."""
        torch = pytest.importorskip("torch")
        from src.features.extractor import FeatureConfig
        from src.policies.mlp_policy import MLPPolicy
        from src.training.train_mlp import PageReplacementMLP

        config = FeatureConfig(use_predicted_reuse=True)
        model = PageReplacementMLP(num_features=config.num_features)
        policy = MLPPolicy(model, config)
        assert policy.feature_config is config
        assert policy.feature_config.num_features == 27
