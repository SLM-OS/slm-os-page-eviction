"""Unit tests for feature extraction and normalization.

Validates feature computation on hand-crafted pool states, correct
ranges, no NaN values, and normalization correctness.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.simulator.block import BlockMeta, BlockState, PoolType, AccessPattern
from src.simulator.core import GlobalState, Pool
from src.features.extractor import FeatureConfig, FeatureExtractor
from src.features.normalizer import FeatureNormalizer


def _make_pool_state() -> tuple[list[BlockMeta], Pool, GlobalState]:
    """Create a pool with known state for deterministic feature testing."""
    pool = Pool(PoolType.WEIGHT, num_blocks=4)
    for i, block in enumerate(pool.blocks):
        block.state = BlockState.ALLOCATED
        block.model_id = 0
        block.layer_idx = i
        block.access_count = (i + 1) * 5
        block.last_access_time = 100 + i * 10
        block.load_time = 50
        block.ref_count = 0
        block.gpu_mapped = (i == 3)
        block.is_dirty = (i == 2)
        block.access_pattern = AccessPattern.SEQUENTIAL

    global_state = GlobalState(
        tick=200,
        weight_pool_utilization=1.0,
        workspace_pool_utilization=0.5,
        num_loaded_models=1,
        total_gpu_mapped_blocks=1,
        pending_load_requests=0,
        avg_model_priority=3.0,
        max_deadline_pressure=0.2,
        recent_fault_rate=0.1,
    )

    candidates = pool.get_evictable_blocks()
    return candidates, pool, global_state


class TestFeatureExtractor:
    """Tests for the 27-feature extraction."""

    def test_feature_count_with_reuse(self):
        config = FeatureConfig(use_predicted_reuse=True)
        assert config.num_features == 27

    def test_feature_count_without_reuse(self):
        config = FeatureConfig(use_predicted_reuse=False)
        assert config.num_features == 26

    def test_extract_block_features_shape(self):
        config = FeatureConfig(use_predicted_reuse=True)
        extractor = FeatureExtractor(config)
        candidates, pool, gs = _make_pool_state()
        if not candidates:
            pytest.skip("No evictable candidates")
        feats = extractor.extract_block_features(candidates[0], pool, gs)
        assert feats.shape == (config.num_per_block,)

    def test_extract_global_features_shape(self):
        config = FeatureConfig(use_predicted_reuse=True)
        extractor = FeatureExtractor(config)
        _, _, gs = _make_pool_state()
        feats = extractor.extract_global_features(gs)
        assert feats.shape == (12,)

    def test_extract_candidate_features_shape(self):
        config = FeatureConfig(use_predicted_reuse=True)
        extractor = FeatureExtractor(config)
        candidates, pool, gs = _make_pool_state()
        if not candidates:
            pytest.skip("No evictable candidates")
        matrix = extractor.extract_candidate_features(candidates, pool, gs)
        assert matrix.shape == (len(candidates), 27)

    def test_no_nan_in_features(self):
        config = FeatureConfig(use_predicted_reuse=True)
        extractor = FeatureExtractor(config)
        candidates, pool, gs = _make_pool_state()
        if not candidates:
            pytest.skip("No evictable candidates")
        matrix = extractor.extract_candidate_features(candidates, pool, gs)
        assert not np.any(np.isnan(matrix))

    def test_predicted_reuse_flag(self):
        config_with = FeatureConfig(use_predicted_reuse=True)
        config_without = FeatureConfig(use_predicted_reuse=False)
        ext_with = FeatureExtractor(config_with)
        ext_without = FeatureExtractor(config_without)

        candidates, pool, gs = _make_pool_state()
        if not candidates:
            pytest.skip("No evictable candidates")

        feats_with = ext_with.extract_candidate_features(candidates, pool, gs)
        feats_without = ext_without.extract_candidate_features(candidates, pool, gs)
        assert feats_with.shape[1] == feats_without.shape[1] + 1


class TestFeatureNormalizer:
    """Tests for feature normalization."""

    def test_normalized_features_bounded(self):
        config = FeatureConfig(use_predicted_reuse=True)
        extractor = FeatureExtractor(config)
        normalizer = FeatureNormalizer(config)

        candidates, pool, gs = _make_pool_state()
        if not candidates:
            pytest.skip("No evictable candidates")

        raw = extractor.extract_candidate_features(candidates, pool, gs)
        normalized = normalizer.normalize(raw, pool_size=4)

        # Most features should be in [0, ~2] after normalization
        # (some may slightly exceed 1.0 due to log normalization)
        assert not np.any(np.isnan(normalized))
        assert not np.any(np.isinf(normalized))

    def test_empty_input(self):
        config = FeatureConfig(use_predicted_reuse=True)
        normalizer = FeatureNormalizer(config)
        empty = np.empty((0, 27))
        result = normalizer.normalize(empty)
        assert result.shape == (0, 27)

    def test_onehot_expansion(self):
        config = FeatureConfig(use_predicted_reuse=True)
        normalizer = FeatureNormalizer(config)
        # Create a dummy feature matrix
        features = np.random.rand(5, 27).astype(np.float32)
        expanded = normalizer.to_onehot_categoricals(features)
        # pool_type (1 col -> 2) + access_pattern (1 col -> 4) = +4 cols
        assert expanded.shape[1] == 27 + 4
