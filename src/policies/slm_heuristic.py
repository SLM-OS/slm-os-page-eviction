"""SLM-aware heuristic eviction policy (Section 5.1 of the plan).

Hand-tuned policy matching the current SLM-OS Phase 3 implementation.
Uses a priority cascade: workspace before weights, inactive models
before active, then LRU as the final tiebreaker.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import EvictionPolicy

if TYPE_CHECKING:
    from src.simulator.block import BlockMeta
    from src.simulator.core import GlobalState, Pool

from src.simulator.block import PoolType


class SLMHeuristicPolicy(EvictionPolicy):
    """Hand-tuned SLM-OS eviction policy.

    Priority cascade:
      1. Evict workspace blocks before weights (cheaper to recreate)
      2. Within workspace, evict LRU
      3. Evict weights from models with no active inferences
      4. Within inactive model weights, evict LRU
      5. Fallback: evict LRU among all remaining candidates
    """

    def __init__(self, active_inferences: dict[int, int] | None = None):
        # Maps model_id -> number of active inference tasks
        self._active_inferences = active_inferences or {}

    def set_active_inferences(self, active: dict[int, int]) -> None:
        """Update the active inference counts (called by simulator)."""
        self._active_inferences = active

    def select_victim(
        self,
        candidates: list[BlockMeta],
        pool: Pool,
        global_state: GlobalState,
    ) -> int:
        # Priority 1: Evict workspace blocks (cheaper to recreate)
        workspace = [
            (i, b) for i, b in enumerate(candidates)
            if b.pool_type == PoolType.WORKSPACE
        ]
        if workspace:
            return self._pick_lru(workspace)

        # Priority 2: Evict weights from models with no active inferences
        inactive = [
            (i, b) for i, b in enumerate(candidates)
            if self._active_inferences.get(b.model_id, 0) == 0
        ]
        if inactive:
            return self._pick_lru(inactive)

        # Priority 3: Evict LRU among remaining
        return self._pick_lru(list(enumerate(candidates)))

    def _pick_lru(self, indexed_blocks: list[tuple[int, BlockMeta]]) -> int:
        """Return the original index of the LRU block."""
        min_time = float("inf")
        victim_idx = indexed_blocks[0][0]
        for idx, block in indexed_blocks:
            if block.last_access_time < min_time:
                min_time = block.last_access_time
                victim_idx = idx
        return victim_idx

    def score(
        self,
        candidates: list[BlockMeta],
        pool: Pool,
        global_state: GlobalState,
    ) -> list[float]:
        """Score candidates by the heuristic priority cascade."""
        scores = [0.0] * len(candidates)
        for i, b in enumerate(candidates):
            score = 0.0
            # Workspace blocks are most evictable
            if b.pool_type == PoolType.WORKSPACE:
                score += 0.6
            # Inactive model blocks are more evictable
            if self._active_inferences.get(b.model_id, 0) == 0:
                score += 0.3
            # Recency tiebreaker (older = more evictable)
            if candidates:
                max_time = max(c.last_access_time for c in candidates)
                min_time = min(c.last_access_time for c in candidates)
                span = max_time - min_time if max_time != min_time else 1
                score += 0.1 * (max_time - b.last_access_time) / span
            scores[i] = score
        return scores

    def name(self) -> str:
        return "SLM-Heuristic"
