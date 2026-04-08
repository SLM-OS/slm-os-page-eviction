"""ARC (Adaptive Replacement Cache) eviction policy.

Maintains ghost lists for recently evicted LRU and LFU candidates,
adapting the balance between recency and frequency based on which
ghost list gets hit more often. Stronger baseline than pure LRU or LFU.

Reference: Megiddo & Modha, "ARC: A Self-Tuning, Low Overhead
Replacement Cache," FAST 2003.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING

from .base import EvictionPolicy

if TYPE_CHECKING:
    from src.simulator.block import BlockMeta
    from src.simulator.core import GlobalState, Pool


class ARCPolicy(EvictionPolicy):
    """Adaptive Replacement Cache with ghost lists.

    Maintains four lists:
      T1: Recently used once (recency)
      T2: Recently used more than once (frequency)
      B1: Ghost list for T1 evictions (recently evicted recency candidates)
      B2: Ghost list for T2 evictions (recently evicted frequency candidates)

    The parameter p controls the split between T1 and T2. When a ghost
    hit occurs in B1, p increases (favor recency). When a ghost hit
    occurs in B2, p decreases (favor frequency).
    """

    def __init__(self, max_ghost_size: int = 256):
        self._t1: OrderedDict[int, int] = OrderedDict()  # block_id -> model_id
        self._t2: OrderedDict[int, int] = OrderedDict()
        self._b1: OrderedDict[int, int] = OrderedDict()  # ghost: block_id -> model_id
        self._b2: OrderedDict[int, int] = OrderedDict()
        self._p: float = 0.0  # Target size for T1
        self._max_ghost = max_ghost_size

    def select_victim(
        self,
        candidates: list[BlockMeta],
        pool: Pool,
        global_state: GlobalState,
    ) -> int:
        # Build a set of candidate IDs for O(1) lookup
        candidate_ids = {b.block_id for b in candidates}

        # Prefer to evict from T1 if T1 is larger than target p
        t1_candidates = [
            i for i, b in enumerate(candidates)
            if b.block_id in self._t1
        ]
        t2_candidates = [
            i for i, b in enumerate(candidates)
            if b.block_id in self._t2
        ]

        if t1_candidates and len(self._t1) > self._p:
            # Evict LRU from T1
            return self._pick_lru(candidates, t1_candidates)
        elif t2_candidates:
            # Evict LRU from T2
            return self._pick_lru(candidates, t2_candidates)
        elif t1_candidates:
            return self._pick_lru(candidates, t1_candidates)
        else:
            # Fallback: evict LRU among all candidates
            return self._pick_lru(candidates, list(range(len(candidates))))

    def _pick_lru(
        self,
        candidates: list[BlockMeta],
        indices: list[int],
    ) -> int:
        """Pick the LRU block among the given candidate indices."""
        min_time = float("inf")
        victim_idx = indices[0]
        for i in indices:
            if candidates[i].last_access_time < min_time:
                min_time = candidates[i].last_access_time
                victim_idx = i
        return victim_idx

    def notify_access(self, block_id: int, model_id: int) -> None:
        """Update ARC state on a block access (called by the simulator).

        This must be called on every access (hit or miss) to maintain
        the T1/T2/B1/B2 lists correctly.
        """
        # Case 1: Hit in T1 — promote to T2
        if block_id in self._t1:
            del self._t1[block_id]
            self._t2[block_id] = model_id
            return

        # Case 2: Hit in T2 — move to MRU of T2
        if block_id in self._t2:
            self._t2.move_to_end(block_id)
            return

        # Case 3: Ghost hit in B1 — increase p (favor recency)
        if block_id in self._b1:
            delta = max(1, len(self._b2) // max(len(self._b1), 1))
            self._p = min(self._p + delta, float(self._max_ghost))
            del self._b1[block_id]
            self._t2[block_id] = model_id
            return

        # Case 4: Ghost hit in B2 — decrease p (favor frequency)
        if block_id in self._b2:
            delta = max(1, len(self._b1) // max(len(self._b2), 1))
            self._p = max(self._p - delta, 0.0)
            del self._b2[block_id]
            self._t2[block_id] = model_id
            return

        # Case 5: Complete miss — add to T1
        self._t1[block_id] = model_id
        self._trim_ghost_lists()

    def notify_eviction(self, block_id: int) -> None:
        """Move an evicted block to the appropriate ghost list."""
        if block_id in self._t1:
            model_id = self._t1.pop(block_id)
            self._b1[block_id] = model_id
        elif block_id in self._t2:
            model_id = self._t2.pop(block_id)
            self._b2[block_id] = model_id

        self._trim_ghost_lists()

    def _trim_ghost_lists(self) -> None:
        """Keep ghost lists within size limits."""
        while len(self._b1) > self._max_ghost:
            self._b1.popitem(last=False)
        while len(self._b2) > self._max_ghost:
            self._b2.popitem(last=False)

    def score(
        self,
        candidates: list[BlockMeta],
        pool: Pool,
        global_state: GlobalState,
    ) -> list[float]:
        """Score candidates for CACHEUS: higher = more desirable to evict."""
        scores = [0.0] * len(candidates)
        for i, b in enumerate(candidates):
            # Blocks in T1 past target size are most evictable
            if b.block_id in self._t1 and len(self._t1) > self._p:
                scores[i] = 0.8
            elif b.block_id in self._t1:
                scores[i] = 0.5
            elif b.block_id in self._t2:
                scores[i] = 0.3
            else:
                # Unknown block — treat as moderately evictable
                scores[i] = 0.6
        # Normalize by recency within tier
        return scores

    def reset(self) -> None:
        self._t1.clear()
        self._t2.clear()
        self._b1.clear()
        self._b2.clear()
        self._p = 0.0

    def name(self) -> str:
        return "ARC"
