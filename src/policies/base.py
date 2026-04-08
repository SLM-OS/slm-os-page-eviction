"""Base eviction policy protocol.

Defines the interface that all eviction policies must implement.
Matches the Rust EvictionPolicy trait from Section 9.1 of the plan.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.simulator.block import BlockMeta
    from src.simulator.core import GlobalState, Pool


class EvictionPolicy(ABC):
    """Abstract base for all eviction policies.

    Each policy must implement select_victim() which chooses an index
    into the evictable candidates list. Policies may optionally implement
    update_feedback() for online learning (e.g., CACHEUS).
    """

    @abstractmethod
    def select_victim(
        self,
        candidates: list[BlockMeta],
        pool: Pool,
        global_state: GlobalState,
    ) -> int:
        """Select the index of the block to evict from the candidates list.

        Args:
            candidates: List of evictable blocks (already filtered by
                        ref_count == 0, not gpu_mapped, not accessing).
            pool: The pool from which the eviction occurs.
            global_state: Current global simulator state.

        Returns:
            Index into the candidates list indicating the chosen victim.
        """
        ...

    def update_feedback(self, block_id: int, was_fault: bool) -> None:
        """Called after an eviction to provide feedback.

        Subclasses that do online learning (e.g., CACHEUS) override this
        to update internal state based on whether the evicted block was
        soon accessed again (fault) or not.

        Args:
            block_id: The block that was evicted.
            was_fault: True if the evicted block was soon re-accessed.
        """

    def reset(self) -> None:
        """Reset internal state for a new simulation run."""

    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this policy."""
        ...

    def score(
        self,
        candidates: list[BlockMeta],
        pool: Pool,
        global_state: GlobalState,
    ) -> list[float]:
        """Score each candidate (higher = more desirable to evict).

        Used by the CACHEUS expert selector. Default implementation returns
        a one-hot vector with 1.0 for the selected victim.
        """
        scores = [0.0] * len(candidates)
        victim_idx = self.select_victim(candidates, pool, global_state)
        scores[victim_idx] = 1.0
        return scores
