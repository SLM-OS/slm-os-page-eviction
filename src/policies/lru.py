"""LRU (Least Recently Used) eviction policy.

Evicts the block with the oldest last_access_time. Effective for
sequential weight sweeps typical of SLM inference. Poor for
frequency-heavy patterns where old but frequently used blocks exist.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import EvictionPolicy

if TYPE_CHECKING:
    from src.simulator.block import BlockMeta
    from src.simulator.core import GlobalState, Pool


class LRUPolicy(EvictionPolicy):
    """Evict the least recently used block."""

    def select_victim(
        self,
        candidates: list[BlockMeta],
        pool: Pool,
        global_state: GlobalState,
    ) -> int:
        min_time = float("inf")
        victim_idx = 0
        for i, block in enumerate(candidates):
            if block.last_access_time < min_time:
                min_time = block.last_access_time
                victim_idx = i
        return victim_idx

    def score(
        self,
        candidates: list[BlockMeta],
        pool: Pool,
        global_state: GlobalState,
    ) -> list[float]:
        """Score by inverse recency: older blocks get higher scores."""
        if not candidates:
            return []
        max_time = max(b.last_access_time for b in candidates)
        min_time = min(b.last_access_time for b in candidates)
        span = max_time - min_time if max_time != min_time else 1
        return [
            (max_time - b.last_access_time) / span
            for b in candidates
        ]

    def name(self) -> str:
        return "LRU"
