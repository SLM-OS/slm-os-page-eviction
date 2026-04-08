"""Unit tests for the Belady optimal oracle.

Validates that Belady always makes the provably correct choice on
traces where the optimal eviction is deterministic.
"""

from __future__ import annotations

import pytest

from src.simulator.block import BlockMeta, BlockState, PoolType, AccessPattern
from src.simulator.core import GlobalState, Pool
from src.simulator.trace import AccessEvent
from src.policies.belady import BeladyOracle


def _make_oracle_scenario() -> (
    tuple[list[BlockMeta], Pool, GlobalState, list[AccessEvent]]
):
    """Create a scenario with known optimal eviction choice.

    Pool has 3 blocks (A, B, C). Future accesses are:
      - A accessed at tick 150
      - B accessed at tick 300
      - C never accessed again

    Optimal eviction: C (furthest/never next access).
    """
    pool = Pool(PoolType.WEIGHT, num_blocks=3)
    for i, block in enumerate(pool.blocks):
        block.state = BlockState.ALLOCATED
        block.model_id = 0
        block.layer_idx = i
        block.last_access_time = 50 + i * 10

    candidates = pool.get_evictable_blocks()
    global_state = GlobalState(tick=100)

    future = [
        AccessEvent(tick=150, block_id=0, model_id=0, layer_idx=0,
                    pool_type=PoolType.WEIGHT, is_hit=True),
        AccessEvent(tick=300, block_id=1, model_id=0, layer_idx=1,
                    pool_type=PoolType.WEIGHT, is_hit=True),
        # Block 2 (layer_idx=2) is never accessed again
    ]

    return candidates, pool, global_state, future


class TestBeladyOracle:
    """Tests for Belady's optimal algorithm."""

    def test_evicts_never_accessed_block(self):
        candidates, pool, gs, future = _make_oracle_scenario()
        oracle = BeladyOracle(future_accesses=future)
        victim_idx = oracle.select_victim(candidates, pool, gs)
        # Block C (layer_idx=2) is never accessed again -> optimal victim
        assert candidates[victim_idx].layer_idx == 2

    def test_evicts_furthest_future_access(self):
        pool = Pool(PoolType.WEIGHT, num_blocks=2)
        for i, block in enumerate(pool.blocks):
            block.state = BlockState.ALLOCATED
            block.model_id = 0
            block.layer_idx = i
        candidates = pool.get_evictable_blocks()
        gs = GlobalState(tick=100)

        # A accessed at 200, B accessed at 500
        future = [
            AccessEvent(tick=200, block_id=0, model_id=0, layer_idx=0,
                        pool_type=PoolType.WEIGHT, is_hit=True),
            AccessEvent(tick=500, block_id=1, model_id=0, layer_idx=1,
                        pool_type=PoolType.WEIGHT, is_hit=True),
        ]
        oracle = BeladyOracle(future_accesses=future)
        victim_idx = oracle.select_victim(candidates, pool, gs)
        # B (layer_idx=1) is further -> evict B
        assert candidates[victim_idx].layer_idx == 1

    def test_reuse_distance_computation(self):
        _, _, gs, future = _make_oracle_scenario()
        oracle = BeladyOracle(future_accesses=future)

        block_a = BlockMeta(
            block_id=0, pool_type=PoolType.WEIGHT,
            state=BlockState.ALLOCATED, model_id=0, layer_idx=0,
        )
        # A is accessed at tick 150, current tick is 100, horizon=1000
        dist = oracle.get_reuse_distance(block_a, current_tick=100, horizon=1000)
        assert abs(dist - 0.05) < 0.001  # (150-100)/1000 = 0.05

    def test_reuse_distance_never_accessed(self):
        _, _, gs, future = _make_oracle_scenario()
        oracle = BeladyOracle(future_accesses=future)

        block_c = BlockMeta(
            block_id=2, pool_type=PoolType.WEIGHT,
            state=BlockState.ALLOCATED, model_id=0, layer_idx=2,
        )
        dist = oracle.get_reuse_distance(block_c, current_tick=100)
        assert dist == 1.0  # Never accessed again

    def test_label_eviction(self):
        candidates, pool, gs, future = _make_oracle_scenario()
        oracle = BeladyOracle(future_accesses=future)
        optimal_idx, reuse_dists = oracle.label_eviction(candidates, pool, gs)
        # Optimal should be the block never accessed again
        assert candidates[optimal_idx].layer_idx == 2
        assert reuse_dists[optimal_idx] == 1.0
        assert len(reuse_dists) == len(candidates)

    def test_tie_breaking_by_cost(self):
        pool = Pool(PoolType.WEIGHT, num_blocks=2)
        for i, block in enumerate(pool.blocks):
            block.state = BlockState.ALLOCATED
            block.model_id = 0
            block.layer_idx = i
        # Both never accessed again
        pool.blocks[0].is_dirty = True   # More expensive to evict
        pool.blocks[1].is_dirty = False  # Cheaper to evict

        candidates = pool.get_evictable_blocks()
        gs = GlobalState(tick=100)
        oracle = BeladyOracle(future_accesses=[])
        victim_idx = oracle.select_victim(candidates, pool, gs)
        # Should prefer cheaper eviction (block 1)
        assert not candidates[victim_idx].is_dirty
