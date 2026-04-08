"""Unit tests for eviction policies.

Validates each policy on simple known-optimal traces where the correct
eviction choice is deterministic.
"""

from __future__ import annotations

import pytest

from src.simulator.block import BlockMeta, BlockState, PoolType, AccessPattern
from src.simulator.core import GlobalState, Pool
from src.policies.lru import LRUPolicy
from src.policies.lfu import LFUPolicy
from src.policies.arc import ARCPolicy
from src.policies.slm_heuristic import SLMHeuristicPolicy


def _make_candidates(
    num: int,
    access_times: list[int] | None = None,
    access_counts: list[int] | None = None,
    pool_types: list[PoolType] | None = None,
) -> tuple[list[BlockMeta], Pool, GlobalState]:
    """Helper to create eviction candidates with specified attributes."""
    pool = Pool(PoolType.WEIGHT, num_blocks=num)
    for i, block in enumerate(pool.blocks):
        block.state = BlockState.ALLOCATED
        block.model_id = 0
        block.layer_idx = i
        if access_times:
            block.last_access_time = access_times[i]
        if access_counts:
            block.access_count = access_counts[i]
        if pool_types:
            block.pool_type = pool_types[i]

    candidates = pool.get_evictable_blocks()
    global_state = GlobalState(tick=1000)
    return candidates, pool, global_state


class TestLRUPolicy:
    """LRU should evict the least recently used block."""

    def test_evicts_oldest(self):
        candidates, pool, gs = _make_candidates(
            3, access_times=[100, 50, 200]
        )
        policy = LRUPolicy()
        victim = policy.select_victim(candidates, pool, gs)
        assert candidates[victim].last_access_time == 50

    def test_single_candidate(self):
        candidates, pool, gs = _make_candidates(1, access_times=[100])
        policy = LRUPolicy()
        assert policy.select_victim(candidates, pool, gs) == 0

    def test_scores_monotonic(self):
        candidates, pool, gs = _make_candidates(
            3, access_times=[100, 50, 200]
        )
        policy = LRUPolicy()
        scores = policy.score(candidates, pool, gs)
        # Oldest (time=50) should have highest score
        assert scores[1] > scores[0] > scores[2]


class TestLFUPolicy:
    """LFU should evict the least frequently used block."""

    def test_evicts_least_accessed(self):
        candidates, pool, gs = _make_candidates(
            3, access_counts=[10, 1, 5]
        )
        policy = LFUPolicy()
        victim = policy.select_victim(candidates, pool, gs)
        assert candidates[victim].access_count == 1

    def test_breaks_ties_by_lru(self):
        candidates, pool, gs = _make_candidates(
            3,
            access_counts=[1, 1, 5],
            access_times=[200, 100, 300],
        )
        policy = LFUPolicy()
        victim = policy.select_victim(candidates, pool, gs)
        # Both block 0 and 1 have count=1; block 1 is older
        assert candidates[victim].last_access_time == 100


class TestARCPolicy:
    """ARC should adapt between recency and frequency."""

    def test_selects_from_candidates(self):
        candidates, pool, gs = _make_candidates(
            4, access_times=[10, 20, 30, 40]
        )
        policy = ARCPolicy()
        victim = policy.select_victim(candidates, pool, gs)
        assert 0 <= victim < len(candidates)

    def test_ghost_list_adaptation(self):
        policy = ARCPolicy()
        # Simulate access pattern that triggers ghost hits
        for i in range(10):
            policy.notify_access(i, model_id=0)
        # Evict some blocks
        for i in range(5):
            policy.notify_eviction(i)
        # Ghost hit in B1 should increase p
        p_before = policy._p
        policy.notify_access(0, model_id=0)  # Ghost hit
        assert policy._p >= p_before


class TestSLMHeuristicPolicy:
    """SLM heuristic should prefer workspace over weights."""

    def test_evicts_workspace_first(self):
        candidates, pool, gs = _make_candidates(
            3,
            access_times=[100, 200, 50],
            pool_types=[PoolType.WEIGHT, PoolType.WORKSPACE, PoolType.WEIGHT],
        )
        policy = SLMHeuristicPolicy()
        victim = policy.select_victim(candidates, pool, gs)
        assert candidates[victim].pool_type == PoolType.WORKSPACE

    def test_evicts_inactive_models_before_active(self):
        candidates, pool, gs = _make_candidates(
            3, access_times=[100, 200, 50]
        )
        candidates[0].model_id = 0
        candidates[1].model_id = 1
        candidates[2].model_id = 0

        policy = SLMHeuristicPolicy(active_inferences={0: 1})  # Model 0 active
        victim = policy.select_victim(candidates, pool, gs)
        # Should evict from model 1 (inactive)
        assert candidates[victim].model_id == 1
