"""Core simulator: memory pools, simulation controller, and event loop.

Implements a discrete-event simulator modeling the SLM-OS ModelAllocator's
two-pool (weight + workspace) memory system with 2MB block granularity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .block import BlockMeta, BlockState, PoolType, AccessPattern

if TYPE_CHECKING:
    from src.policies.base import EvictionPolicy
    from .trace import TraceCollector
    from .workload import AccessRequest


@dataclass
class PoolStats:
    """Runtime statistics for a memory pool."""
    total_blocks: int = 0
    free_blocks: int = 0
    allocated_blocks: int = 0
    shared_blocks: int = 0      # Blocks with ref_count > 1
    gpu_mapped_blocks: int = 0
    peak_usage: int = 0


class Pool:
    """A fixed-size pool of 2MB memory blocks (weight or workspace).

    Maintains block metadata and supports allocation, freeing, access,
    and eviction operations matching the SLM-OS ModelAllocator semantics.
    """

    def __init__(self, pool_type: PoolType, num_blocks: int):
        self.pool_type = pool_type
        self.num_blocks = num_blocks
        self.blocks: list[BlockMeta] = [
            BlockMeta(block_id=i, pool_type=pool_type)
            for i in range(num_blocks)
        ]
        self._next_id_offset = 0  # For global unique IDs across pools

    def set_id_offset(self, offset: int) -> None:
        """Set a global ID offset so block IDs are unique across pools."""
        for i, block in enumerate(self.blocks):
            block.block_id = offset + i
        self._next_id_offset = offset

    def find_free_block(self) -> BlockMeta | None:
        """Return the first free block, or None if pool is full."""
        for block in self.blocks:
            if block.state == BlockState.FREE:
                return block
        return None

    def get_evictable_blocks(self) -> list[BlockMeta]:
        """Return all blocks eligible for eviction (Section 3.2 of the plan)."""
        return [b for b in self.blocks if b.is_evictable()]

    def get_allocated_blocks(self) -> list[BlockMeta]:
        """Return all blocks currently allocated (not free)."""
        return [b for b in self.blocks if b.state != BlockState.FREE]

    def get_stats(self) -> PoolStats:
        """Compute current pool statistics."""
        stats = PoolStats(total_blocks=self.num_blocks)
        for block in self.blocks:
            if block.state == BlockState.FREE:
                stats.free_blocks += 1
            else:
                stats.allocated_blocks += 1
                if block.ref_count > 1:
                    stats.shared_blocks += 1
                if block.gpu_mapped:
                    stats.gpu_mapped_blocks += 1
        stats.peak_usage = max(
            stats.peak_usage, stats.allocated_blocks
        )
        return stats

    def recency_rank(self, target: BlockMeta) -> int:
        """Rank of target block by last access time (0 = most recent)."""
        allocated = self.get_allocated_blocks()
        sorted_blocks = sorted(allocated, key=lambda b: b.last_access_time, reverse=True)
        for rank, block in enumerate(sorted_blocks):
            if block.block_id == target.block_id:
                return rank
        return len(allocated) - 1

    def frequency_rank(self, target: BlockMeta) -> int:
        """Rank of target block by access count (0 = most accessed)."""
        allocated = self.get_allocated_blocks()
        sorted_blocks = sorted(allocated, key=lambda b: b.access_count, reverse=True)
        for rank, block in enumerate(sorted_blocks):
            if block.block_id == target.block_id:
                return rank
        return len(allocated) - 1


class MemoryState:
    """Combined weight + workspace pool state for the simulator.

    Models the dual-pool architecture of SLM-OS's ModelAllocator.
    """

    def __init__(self, weight_blocks: int = 64, workspace_blocks: int = 32):
        self.weight_pool = Pool(PoolType.WEIGHT, weight_blocks)
        self.workspace_pool = Pool(PoolType.WORKSPACE, workspace_blocks)
        # Ensure unique block IDs across both pools
        self.workspace_pool.set_id_offset(weight_blocks)

    def get_pool(self, pool_type: PoolType) -> Pool:
        """Return the pool for the given type."""
        if pool_type == PoolType.WEIGHT:
            return self.weight_pool
        return self.workspace_pool

    def get_block_by_id(self, block_id: int) -> BlockMeta | None:
        """Look up a block by its global ID across both pools."""
        for pool in (self.weight_pool, self.workspace_pool):
            for block in pool.blocks:
                if block.block_id == block_id:
                    return block
        return None

    @property
    def total_blocks(self) -> int:
        return self.weight_pool.num_blocks + self.workspace_pool.num_blocks


@dataclass
class GlobalState:
    """Global simulator state exposed to eviction policies (Section 2.2)."""
    tick: int = 0
    weight_pool_utilization: float = 0.0
    workspace_pool_utilization: float = 0.0
    num_loaded_models: int = 0
    total_gpu_mapped_blocks: int = 0
    pending_load_requests: int = 0
    avg_model_priority: float = 0.0
    max_deadline_pressure: float = 0.0
    recent_fault_rate: float = 0.0
    hot_swap_in_progress: bool = False
    # Requesting block context
    requesting_block_pool: PoolType = PoolType.WEIGHT
    requesting_block_model_id: int = 0
    requesting_block_priority: int = 0


class SimController:
    """Discrete-event simulation controller.

    Drives the simulation loop: processes access requests from workloads,
    triggers evictions via the configured policy when pools are full,
    and records events through the trace collector.
    """

    def __init__(
        self,
        memory: MemoryState,
        policy: EvictionPolicy,
        trace_collector: TraceCollector | None = None,
    ):
        self.memory = memory
        self.policy = policy
        self.trace = trace_collector
        self.tick: int = 0
        self.total_accesses: int = 0
        self.total_faults: int = 0
        self.total_evictions: int = 0
        self._loaded_models: dict[int, set[int]] = {}  # model_id -> set of block_ids
        self._model_priorities: dict[int, int] = {}     # model_id -> priority
        self._active_inferences: dict[int, int] = {}    # model_id -> active count
        self._recent_accesses: int = 0
        self._recent_faults: int = 0
        self._fault_window: int = 100  # Ticks for recent_fault_rate
        # Track recently evicted content for CACHEUS feedback
        self._evicted_content: dict[tuple[int, int, int], tuple[int, int]] = {}
        # Key: (model_id, layer_idx, pool_type) -> (block_id, eviction_tick)
        self._eviction_feedback_window: int = 200  # Ticks before declaring "good eviction"

    def get_global_state(
        self,
        requesting_pool: PoolType = PoolType.WEIGHT,
        requesting_model_id: int = 0,
        requesting_priority: int = 0,
    ) -> GlobalState:
        """Compute the current global state snapshot for policy decisions."""
        w_stats = self.memory.weight_pool.get_stats()
        ws_stats = self.memory.workspace_pool.get_stats()

        priorities = list(self._model_priorities.values())
        avg_priority = sum(priorities) / len(priorities) if priorities else 0.0

        return GlobalState(
            tick=self.tick,
            weight_pool_utilization=(
                w_stats.allocated_blocks / w_stats.total_blocks
                if w_stats.total_blocks > 0 else 0.0
            ),
            workspace_pool_utilization=(
                ws_stats.allocated_blocks / ws_stats.total_blocks
                if ws_stats.total_blocks > 0 else 0.0
            ),
            num_loaded_models=len(self._loaded_models),
            total_gpu_mapped_blocks=(
                w_stats.gpu_mapped_blocks + ws_stats.gpu_mapped_blocks
            ),
            pending_load_requests=0,  # Updated during simulation
            avg_model_priority=avg_priority,
            max_deadline_pressure=0.0,  # Updated from scheduler context
            recent_fault_rate=(
                self._recent_faults / self._recent_accesses
                if self._recent_accesses > 0 else 0.0
            ),
            hot_swap_in_progress=False,  # Updated during hot-swap scenarios
            requesting_block_pool=requesting_pool,
            requesting_block_model_id=requesting_model_id,
            requesting_block_priority=requesting_priority,
        )

    def process_access(self, request: AccessRequest) -> bool:
        """Process a single memory access request.

        Returns True if the access was a hit (block already loaded),
        False if it caused a fault (eviction + reload needed).
        """
        self.tick = request.tick
        self.total_accesses += 1
        self._recent_accesses += 1

        pool = self.memory.get_pool(request.pool_type)

        # Check if the block is already loaded (hit)
        for block in pool.blocks:
            if (
                block.state != BlockState.FREE
                and block.model_id == request.model_id
                and block.layer_idx == request.layer_idx
            ):
                block.record_access(self.tick)
                if self.trace:
                    self.trace.record_access(
                        tick=self.tick,
                        block_id=block.block_id,
                        model_id=request.model_id,
                        layer_idx=request.layer_idx,
                        pool_type=request.pool_type,
                        is_hit=True,
                    )
                return True

        # Miss -- need to load the block
        self.total_faults += 1
        self._recent_faults += 1

        # Try to find a free block first
        block = pool.find_free_block()

        if block is None:
            # Pool full -- must evict
            evictable = pool.get_evictable_blocks()
            if not evictable:
                # No evictable blocks; in real SLM-OS this would block.
                # In simulation, skip this request.
                return False

            global_state = self.get_global_state(
                requesting_pool=request.pool_type,
                requesting_model_id=request.model_id,
                requesting_priority=request.priority,
            )

            victim_idx = self.policy.select_victim(evictable, pool, global_state)
            victim = evictable[victim_idx]

            if self.trace:
                self.trace.record_eviction(
                    tick=self.tick,
                    victim_block_id=victim.block_id,
                    victim_model_id=victim.model_id,
                    victim_layer_idx=victim.layer_idx,
                    pool_type=request.pool_type,
                    evictable_blocks=evictable,
                    global_state=global_state,
                )

            # Remove from model tracking
            if victim.model_id in self._loaded_models:
                self._loaded_models[victim.model_id].discard(victim.block_id)
                if not self._loaded_models[victim.model_id]:
                    del self._loaded_models[victim.model_id]

            # Track evicted content for CACHEUS feedback
            evict_key = (victim.model_id, victim.layer_idx, int(victim.pool_type))
            self._evicted_content[evict_key] = (victim.block_id, self.tick)

            victim.reset()
            block = victim
            self.total_evictions += 1

        # Load into the free/evicted block
        block.state = BlockState.ALLOCATED
        block.model_id = request.model_id
        block.layer_idx = request.layer_idx
        block.pool_type = request.pool_type
        block.load_time = self.tick
        block.record_access(self.tick)
        block.access_pattern = request.access_pattern
        block.ref_count = 0  # Block is accessible but not held; evictable after load

        # Track model membership
        if request.model_id not in self._loaded_models:
            self._loaded_models[request.model_id] = set()
        self._loaded_models[request.model_id].add(block.block_id)
        self._model_priorities[request.model_id] = request.priority

        if self.trace:
            self.trace.record_access(
                tick=self.tick,
                block_id=block.block_id,
                model_id=request.model_id,
                layer_idx=request.layer_idx,
                pool_type=request.pool_type,
                is_hit=False,
            )

        # Check if this miss is a re-access of recently evicted content (bad eviction)
        content_key = (request.model_id, request.layer_idx, int(request.pool_type))
        if content_key in self._evicted_content:
            evicted_block_id, evict_tick = self._evicted_content.pop(content_key)
            self.policy.update_feedback(evicted_block_id, was_fault=True)

        # Flush old eviction records: content not re-accessed within window = good eviction
        stale_keys = [
            k for k, (bid, t) in self._evicted_content.items()
            if self.tick - t > self._eviction_feedback_window
        ]
        for k in stale_keys:
            bid, _ = self._evicted_content.pop(k)
            self.policy.update_feedback(bid, was_fault=False)

        return False

    def run(self, requests: list[AccessRequest]) -> None:
        """Run the full simulation over a sequence of access requests."""
        # Reset fault window tracking periodically
        window_start = 0
        for request in requests:
            if request.tick - window_start >= self._fault_window:
                self._recent_accesses = 0
                self._recent_faults = 0
                window_start = request.tick
            self.process_access(request)

    def get_active_inferences(self, model_id: int) -> int:
        """Return the number of active inference tasks for a model."""
        return self._active_inferences.get(model_id, 0)

    def start_inference(self, model_id: int) -> None:
        """Mark an inference task as active for a model."""
        self._active_inferences[model_id] = (
            self._active_inferences.get(model_id, 0) + 1
        )

    def end_inference(self, model_id: int) -> None:
        """Mark an inference task as completed for a model."""
        if model_id in self._active_inferences:
            self._active_inferences[model_id] -= 1
            if self._active_inferences[model_id] <= 0:
                del self._active_inferences[model_id]
