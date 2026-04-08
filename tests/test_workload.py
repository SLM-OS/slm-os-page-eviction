"""Unit tests for the workload generator.

Validates trace length, access pattern correctness, model identity,
and reproducibility with seeds.
"""

from __future__ import annotations

import pytest

from src.simulator.block import PoolType, AccessPattern
from src.simulator.workload import (
    WorkloadGenerator,
    ModelConfig,
    MODELS,
    AccessRequest,
)


class TestModelConfig:
    """Tests for model archetypes."""

    def test_all_models_defined(self):
        assert len(MODELS) == 6
        for name in ["tiny", "small", "medium", "large", "attn", "critical"]:
            assert name in MODELS

    def test_model_ids_unique(self):
        ids = [m.model_id for m in MODELS.values()]
        assert len(ids) == len(set(ids))

    def test_critical_has_highest_priority(self):
        assert MODELS["critical"].priority == 7
        assert all(
            m.priority <= 7 for m in MODELS.values()
        )


class TestWorkloadGenerator:
    """Tests for scenario generation."""

    def test_reproducibility(self):
        wg1 = WorkloadGenerator(seed=42)
        wg2 = WorkloadGenerator(seed=42)
        r1 = wg1.gen_single_inference(MODELS["tiny"])
        r2 = wg2.gen_single_inference(MODELS["tiny"])
        assert len(r1) == len(r2)
        for a, b in zip(r1, r2):
            assert a.tick == b.tick
            assert a.model_id == b.model_id
            assert a.layer_idx == b.layer_idx

    def test_single_inference_produces_requests(self):
        wg = WorkloadGenerator(seed=42)
        requests = wg.gen_single_inference(MODELS["tiny"], num_inferences=2)
        assert len(requests) > 0
        assert all(isinstance(r, AccessRequest) for r in requests)

    def test_single_inference_has_weight_and_workspace(self):
        wg = WorkloadGenerator(seed=42)
        requests = wg.gen_single_inference(MODELS["small"], num_inferences=1)
        pool_types = {r.pool_type for r in requests}
        assert PoolType.WEIGHT in pool_types
        assert PoolType.WORKSPACE in pool_types

    def test_multi_model_interleaves(self):
        wg = WorkloadGenerator(seed=42)
        requests = wg.gen_multi_model()
        model_ids = {r.model_id for r in requests}
        assert len(model_ids) > 1  # Multiple models present
        # Verify sorted by tick
        for i in range(1, len(requests)):
            assert requests[i].tick >= requests[i - 1].tick

    def test_hot_swap_has_two_models(self):
        wg = WorkloadGenerator(seed=42)
        requests = wg.gen_hot_swap()
        model_ids = {r.model_id for r in requests}
        assert len(model_ids) == 2

    def test_mixed_priority_has_high_and_low(self):
        wg = WorkloadGenerator(seed=42)
        requests = wg.gen_mixed_priority()
        priorities = {r.priority for r in requests}
        assert max(priorities) > min(priorities)

    def test_gpu_contention_marks_gpu(self):
        wg = WorkloadGenerator(seed=42)
        requests = wg.gen_gpu_contention()
        assert any(r.is_gpu for r in requests)

    def test_adversarial_exceeds_pool_size(self):
        wg = WorkloadGenerator(seed=42)
        pool_size = 8
        requests = wg.gen_adversarial(pool_size=pool_size, num_accesses=100)
        unique_layers = {r.layer_idx for r in requests}
        assert len(unique_layers) > pool_size

    def test_all_scenarios_available(self):
        names = WorkloadGenerator.all_scenario_names()
        assert len(names) == 7
        wg = WorkloadGenerator(seed=42)
        for name in names:
            requests = wg.generate_scenario(name)
            assert len(requests) > 0

    def test_unknown_scenario_raises(self):
        wg = WorkloadGenerator(seed=42)
        with pytest.raises(ValueError, match="Unknown scenario"):
            wg.generate_scenario("nonexistent")
