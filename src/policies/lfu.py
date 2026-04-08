"""LFU (Least Frequently Used) eviction policy.

Evicts the block with the lowest access_count. Effective for long-running
models with stable hot sets. Poor for newly-loaded blocks (cold start
problem) since new blocks always have low counts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import EvictionPolicy

if TYPE_CHECKING:
    from src.simulator.block import BlockMeta
    from src.simulator.core import GlobalState, Pool


class LFUPolicy(EvictionPolicy):
    """Evict the least frequently used block."""

    def select_victim(
        self,
        candidates: list[BlockMeta],
        pool: Pool,
        global_state: GlobalState,
    ) -> int:
        min_count = float("inf")
        victim_idx = 0
        for i, block in enumerate(candidates):
            if block.access_count < min_count:
                min_count = block.access_count
                victim_idx = i
            elif block.access_count == min_count:
                # Break ties by LRU (older block evicted first)
                if block.last_access_time < candidates[victim_idx].last_access_time:
                    victim_idx = i
        return victim_idx

    def score(
        self,
        candidates: list[BlockMeta],
        pool: Pool,
        global_state: GlobalState,
    ) -> list[float]:
        """Score by inverse frequency: less-accessed blocks get higher scores."""
        if not candidates:
            return []
        max_count = max(b.access_count for b in candidates)
        min_count = min(b.access_count for b in candidates)
        span = max_count - min_count if max_count != min_count else 1
        return [
            (max_count - b.access_count) / span
            for b in candidates
        ]

    def name(self) -> str:
        return "LFU"
