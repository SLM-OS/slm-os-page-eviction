"""Belady's optimal offline oracle (Section 4 of the plan).

Computes the provably optimal eviction decision given complete knowledge
of future accesses. Used to generate training labels (is_optimal) and
as an upper bound for policy comparison.

This oracle requires the full access trace to be available, so it can
only run offline (not at runtime in SLM-OS).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import EvictionPolicy

if TYPE_CHECKING:
    from src.simulator.block import BlockMeta
    from src.simulator.core import GlobalState, Pool
    from src.simulator.trace import AccessEvent


class BeladyOracle(EvictionPolicy):
    """Belady's MIN algorithm: evict the block used furthest in the future.

    Given the complete sequence of future accesses, this always makes the
    optimal eviction choice. Ties (multiple blocks never accessed again)
    are broken by eviction cost (prefer evicting cheap-to-reload blocks).
    """

    def __init__(self, future_accesses: list[AccessEvent] | None = None):
        self._future_accesses: list[AccessEvent] = future_accesses or []
        self._access_index: int = 0  # Current position in the trace

    def set_future_accesses(self, accesses: list[AccessEvent]) -> None:
        """Set the complete future access trace for oracle decisions."""
        self._future_accesses = accesses
        self._access_index = 0

    def select_victim(
        self,
        candidates: list[BlockMeta],
        pool: Pool,
        global_state: GlobalState,
    ) -> int:
        """Select the candidate whose next access is furthest in the future."""
        current_tick = global_state.tick

        # For each candidate, find the next time it will be accessed
        next_access_tick: dict[int, float] = {}
        for block in candidates:
            next_access_tick[block.block_id] = float("inf")
            for event in self._future_accesses:
                if event.tick <= current_tick:
                    continue
                if self._event_targets_block(event, block):
                    next_access_tick[block.block_id] = event.tick
                    break

        # Evict the block with the furthest next access
        max_next = -1.0
        victim_idx = 0
        for i, block in enumerate(candidates):
            dist = next_access_tick[block.block_id]
            if dist > max_next:
                max_next = dist
                victim_idx = i
            elif dist == max_next:
                # Tie-break: prefer cheaper eviction (not dirty, not GPU)
                current_victim = candidates[victim_idx]
                if self._eviction_cost(block) < self._eviction_cost(current_victim):
                    victim_idx = i

        return victim_idx

    def get_reuse_distance(
        self,
        block: BlockMeta,
        current_tick: int,
        horizon: int = 1000,
    ) -> float:
        """Compute normalized reuse distance for a block.

        Returns the number of ticks until the next access, normalized
        by the horizon window. Returns 1.0 if the block is never
        accessed again within the horizon.
        """
        for event in self._future_accesses:
            if event.tick <= current_tick:
                continue
            if self._event_targets_block(event, block):
                distance = event.tick - current_tick
                return min(distance / horizon, 1.0)
        return 1.0  # Never accessed again

    def label_eviction(
        self,
        candidates: list[BlockMeta],
        pool: Pool,
        global_state: GlobalState,
    ) -> tuple[int, list[float]]:
        """Label an eviction event with optimal choice and reuse distances.

        Returns:
            Tuple of (optimal_index, reuse_distances) where:
              - optimal_index is the Belady-optimal victim
              - reuse_distances[i] is the normalized reuse distance for candidate i
        """
        optimal_idx = self.select_victim(candidates, pool, global_state)
        reuse_distances = [
            self.get_reuse_distance(b, global_state.tick)
            for b in candidates
        ]
        return optimal_idx, reuse_distances

    @staticmethod
    def _event_targets_block(event: AccessEvent, block: BlockMeta) -> bool:
        """Check if an access event targets the given block's content."""
        return (
            event.model_id == block.model_id
            and event.layer_idx == block.layer_idx
            and int(event.pool_type) == int(block.pool_type)
        )

    @staticmethod
    def _eviction_cost(block: BlockMeta) -> float:
        """Estimate eviction cost for tie-breaking."""
        cost = 1.0
        if block.is_dirty:
            cost += 0.3
        if block.gpu_mapped:
            cost += 0.5
        return cost

    def score(
        self,
        candidates: list[BlockMeta],
        pool: Pool,
        global_state: GlobalState,
    ) -> list[float]:
        """Score by reuse distance (higher distance = more evictable)."""
        return [
            self.get_reuse_distance(b, global_state.tick)
            for b in candidates
        ]

    def reset(self) -> None:
        self._access_index = 0

    def name(self) -> str:
        return "Belady-Optimal"
