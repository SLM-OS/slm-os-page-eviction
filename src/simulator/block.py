"""Block metadata, pool types, and access patterns for the SLM-OS memory simulator.

Models the 2MB block granularity of the ModelAllocator in runtime/src/mm/model_mem.rs.
Each block tracks access history, ownership, and state needed for eviction decisions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum


class PoolType(IntEnum):
    """Memory pool type, matching SLM-OS weight/workspace distinction."""
    WEIGHT = 0      # Read-only model weights (shared across tasks)
    WORKSPACE = 1   # Read-write scratch space (per-task)


class AccessPattern(IntEnum):
    """Observed access pattern for a block."""
    SEQUENTIAL = 0  # Layer-by-layer sweep (typical weight access)
    RANDOM = 1      # Unpredictable access (rare for SLM workloads)
    STRIDED = 2     # Skip-layer patterns (attention models)
    BURST = 3       # Rapid repeated access (hot workspace)


class BlockState(IntEnum):
    """Lifecycle state of a memory block."""
    FREE = 0        # Available for allocation
    ALLOCATED = 1   # In use, may be evictable
    ACCESSING = 2   # Currently being read/written (not evictable)
    EVICTING = 3    # Being evicted (writeback in progress)


@dataclass
class BlockMeta:
    """Metadata for a single 2MB memory block.

    Tracks all state needed for eviction policy decisions: access history,
    ownership, GPU mapping, and pool membership.
    """
    block_id: int
    pool_type: PoolType
    state: BlockState = BlockState.FREE

    # Ownership
    model_id: int = -1          # Owning model (-1 = unassigned)
    layer_idx: int = -1         # Layer index within model (-1 = workspace)
    owner_task: int = -1        # Task that allocated this block

    # Access tracking
    access_count: int = 0       # Total accesses since load
    last_access_time: int = 0   # Tick of most recent access
    load_time: int = 0          # Tick when block was loaded into pool
    access_pattern: AccessPattern = AccessPattern.SEQUENTIAL

    # Sharing and mapping
    ref_count: int = 0          # Number of tasks sharing this block
    gpu_mapped: bool = False    # Currently accessible by GPU
    is_dirty: bool = False      # Modified since load (writeback cost)

    # Derived (set by feature extractor, not stored persistently)
    generation: int = 0         # Incremented on reuse for stale detection

    def is_evictable(self) -> bool:
        """A block can be evicted only if unreferenced, not GPU-mapped,
        and not currently being accessed."""
        return (
            self.ref_count == 0
            and not self.gpu_mapped
            and self.state not in (BlockState.FREE, BlockState.ACCESSING, BlockState.EVICTING)
        )

    def record_access(self, tick: int) -> None:
        """Update access tracking on a new access event."""
        self.access_count += 1
        self.last_access_time = tick

    def reset(self) -> None:
        """Clear block metadata for reuse after eviction."""
        self.state = BlockState.FREE
        self.model_id = -1
        self.layer_idx = -1
        self.owner_task = -1
        self.access_count = 0
        self.last_access_time = 0
        self.load_time = 0
        self.access_pattern = AccessPattern.SEQUENTIAL
        self.ref_count = 0
        self.gpu_mapped = False
        self.is_dirty = False
        self.generation += 1
