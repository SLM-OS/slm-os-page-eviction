"""Unit tests for the simulator core: MemoryState, Pool, SimController.

Tests alloc/free cycles, ref_count semantics, pool capacity limits,
and the eviction trigger path.
"""

from __future__ import annotations

import pytest

from src.simulator.block import BlockMeta, BlockState, PoolType, AccessPattern
from src.simulator.core import MemoryState, Pool, SimController, GlobalState
from src.simulator.trace import TraceCollector
from src.policies.lru import LRUPolicy


class TestBlockMeta:
    """Tests for BlockMeta data class."""

    def test_new_block_is_free(self):
        block = BlockMeta(block_id=0, pool_type=PoolType.WEIGHT)
        assert block.state == BlockState.FREE
        assert block.access_count == 0
        assert block.ref_count == 0

    def test_evictable_requires_allocated(self):
        block = BlockMeta(block_id=0, pool_type=PoolType.WEIGHT)
        assert not block.is_evictable()  # FREE is not evictable

        block.state = BlockState.ALLOCATED
        assert block.is_evictable()

    def test_not_evictable_if_referenced(self):
        block = BlockMeta(
            block_id=0, pool_type=PoolType.WEIGHT,
            state=BlockState.ALLOCATED, ref_count=1,
        )
        assert not block.is_evictable()

    def test_not_evictable_if_gpu_mapped(self):
        block = BlockMeta(
            block_id=0, pool_type=PoolType.WEIGHT,
            state=BlockState.ALLOCATED, gpu_mapped=True,
        )
        assert not block.is_evictable()

    def test_not_evictable_if_accessing(self):
        block = BlockMeta(
            block_id=0, pool_type=PoolType.WEIGHT,
            state=BlockState.ACCESSING,
        )
        assert not block.is_evictable()

    def test_record_access_updates_state(self):
        block = BlockMeta(block_id=0, pool_type=PoolType.WEIGHT)
        block.record_access(tick=100)
        assert block.access_count == 1
        assert block.last_access_time == 100

        block.record_access(tick=200)
        assert block.access_count == 2
        assert block.last_access_time == 200

    def test_reset_clears_all_state(self):
        block = BlockMeta(
            block_id=5, pool_type=PoolType.WEIGHT,
            state=BlockState.ALLOCATED, model_id=2,
            access_count=10, ref_count=1, gpu_mapped=True,
        )
        gen = block.generation
        block.reset()
        assert block.state == BlockState.FREE
        assert block.model_id == -1
        assert block.access_count == 0
        assert block.ref_count == 0
        assert not block.gpu_mapped
        assert block.generation == gen + 1


class TestPool:
    """Tests for Pool operations."""

    def test_new_pool_all_free(self):
        pool = Pool(PoolType.WEIGHT, num_blocks=8)
        assert pool.find_free_block() is not None
        assert len(pool.get_evictable_blocks()) == 0  # FREE is not evictable
        stats = pool.get_stats()
        assert stats.free_blocks == 8
        assert stats.allocated_blocks == 0

    def test_find_free_block_returns_none_when_full(self):
        pool = Pool(PoolType.WEIGHT, num_blocks=2)
        for block in pool.blocks:
            block.state = BlockState.ALLOCATED
        assert pool.find_free_block() is None

    def test_evictable_blocks_filtered(self):
        pool = Pool(PoolType.WEIGHT, num_blocks=4)
        pool.blocks[0].state = BlockState.ALLOCATED  # evictable
        pool.blocks[1].state = BlockState.ALLOCATED
        pool.blocks[1].ref_count = 1  # not evictable
        pool.blocks[2].state = BlockState.ACCESSING  # not evictable
        pool.blocks[3].state = BlockState.FREE  # not evictable
        evictable = pool.get_evictable_blocks()
        assert len(evictable) == 1
        assert evictable[0].block_id == 0

    def test_unique_ids_across_pools(self):
        pool1 = Pool(PoolType.WEIGHT, num_blocks=4)
        pool2 = Pool(PoolType.WORKSPACE, num_blocks=4)
        pool2.set_id_offset(4)
        ids = {b.block_id for b in pool1.blocks} | {b.block_id for b in pool2.blocks}
        assert len(ids) == 8

    def test_recency_rank(self):
        pool = Pool(PoolType.WEIGHT, num_blocks=3)
        for i, block in enumerate(pool.blocks):
            block.state = BlockState.ALLOCATED
            block.last_access_time = (i + 1) * 10
        # Block 2 (time=30) is most recent -> rank 0
        assert pool.recency_rank(pool.blocks[2]) == 0
        # Block 0 (time=10) is least recent -> rank 2
        assert pool.recency_rank(pool.blocks[0]) == 2


class TestMemoryState:
    """Tests for dual-pool MemoryState."""

    def test_block_ids_unique(self):
        mem = MemoryState(weight_blocks=4, workspace_blocks=4)
        all_ids = set()
        for pool in (mem.weight_pool, mem.workspace_pool):
            for block in pool.blocks:
                assert block.block_id not in all_ids
                all_ids.add(block.block_id)

    def test_get_block_by_id(self):
        mem = MemoryState(weight_blocks=4, workspace_blocks=4)
        block = mem.get_block_by_id(0)
        assert block is not None
        assert block.pool_type == PoolType.WEIGHT

        block = mem.get_block_by_id(4)
        assert block is not None
        assert block.pool_type == PoolType.WORKSPACE

    def test_total_blocks(self):
        mem = MemoryState(weight_blocks=8, workspace_blocks=4)
        assert mem.total_blocks == 12


class TestSimController:
    """Tests for the simulation controller."""

    def test_hit_does_not_fault(self):
        from src.simulator.workload import AccessRequest
        mem = MemoryState(weight_blocks=4, workspace_blocks=2)
        policy = LRUPolicy()
        sim = SimController(memory=mem, policy=policy)

        # First access: miss (cold start)
        req = AccessRequest(
            tick=1, model_id=0, layer_idx=0,
            pool_type=PoolType.WEIGHT,
            access_pattern=AccessPattern.SEQUENTIAL,
        )
        hit = sim.process_access(req)
        assert not hit
        assert sim.total_faults == 1

        # Second access to same block: hit
        req2 = AccessRequest(
            tick=2, model_id=0, layer_idx=0,
            pool_type=PoolType.WEIGHT,
            access_pattern=AccessPattern.SEQUENTIAL,
        )
        hit = sim.process_access(req2)
        assert hit
        assert sim.total_faults == 1  # No new fault

    def test_eviction_when_pool_full(self):
        from src.simulator.workload import AccessRequest
        mem = MemoryState(weight_blocks=2, workspace_blocks=1)
        policy = LRUPolicy()
        trace = TraceCollector()
        sim = SimController(memory=mem, policy=policy, trace_collector=trace)

        # Fill the pool
        for i in range(2):
            sim.process_access(AccessRequest(
                tick=i, model_id=0, layer_idx=i,
                pool_type=PoolType.WEIGHT,
                access_pattern=AccessPattern.SEQUENTIAL,
            ))

        # This should trigger an eviction
        sim.process_access(AccessRequest(
            tick=10, model_id=0, layer_idx=2,
            pool_type=PoolType.WEIGHT,
            access_pattern=AccessPattern.SEQUENTIAL,
        ))

        assert sim.total_evictions == 1
        assert trace.num_evictions == 1
