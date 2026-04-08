"""Trace collector: records access and eviction events for analysis.

Captures a complete log of memory operations during simulation, including
which blocks were accessed, which were evicted, and the state at each
eviction decision point. Exports to Parquet for dataset generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .block import BlockMeta, PoolType

if TYPE_CHECKING:
    from .core import GlobalState


@dataclass
class AccessEvent:
    """A single memory access (hit or miss)."""
    tick: int
    block_id: int
    model_id: int
    layer_idx: int
    pool_type: PoolType
    is_hit: bool


@dataclass
class EvictionEvent:
    """A single eviction decision with full context.

    Records the victim chosen, all candidates considered, and the global
    state at the time of the decision. Used for Belady labeling and
    feature extraction.
    """
    tick: int
    eviction_id: int
    victim_block_id: int
    victim_model_id: int
    victim_layer_idx: int
    pool_type: PoolType
    num_candidates: int
    candidate_block_ids: list[int]
    global_state: GlobalState


class TraceCollector:
    """Collects access and eviction events during simulation.

    Maintains two event logs that can be exported to Parquet for
    offline analysis, Belady labeling, and dataset generation.
    """

    def __init__(self, scenario: str = "", seed: int = 0):
        self.scenario = scenario
        self.seed = seed
        self.access_events: list[AccessEvent] = []
        self.eviction_events: list[EvictionEvent] = []
        self._next_eviction_id: int = 0

    def record_access(
        self,
        tick: int,
        block_id: int,
        model_id: int,
        layer_idx: int,
        pool_type: PoolType,
        is_hit: bool,
    ) -> None:
        """Record a memory access event."""
        self.access_events.append(AccessEvent(
            tick=tick,
            block_id=block_id,
            model_id=model_id,
            layer_idx=layer_idx,
            pool_type=pool_type,
            is_hit=is_hit,
        ))

    def record_eviction(
        self,
        tick: int,
        victim_block_id: int,
        victim_model_id: int,
        victim_layer_idx: int,
        pool_type: PoolType,
        evictable_blocks: list[BlockMeta],
        global_state: GlobalState,
    ) -> None:
        """Record an eviction decision with full context."""
        self.eviction_events.append(EvictionEvent(
            tick=tick,
            eviction_id=self._next_eviction_id,
            victim_block_id=victim_block_id,
            victim_model_id=victim_model_id,
            victim_layer_idx=victim_layer_idx,
            pool_type=pool_type,
            num_candidates=len(evictable_blocks),
            candidate_block_ids=[b.block_id for b in evictable_blocks],
            global_state=global_state,
        ))
        self._next_eviction_id += 1

    def get_future_accesses(self, from_tick: int) -> list[AccessEvent]:
        """Return all access events after the given tick.

        Used by the Belady oracle to compute optimal eviction decisions.
        """
        return [e for e in self.access_events if e.tick > from_tick]

    def to_access_dataframe(self) -> pd.DataFrame:
        """Export access events as a pandas DataFrame."""
        if not self.access_events:
            return pd.DataFrame()

        records = []
        for e in self.access_events:
            records.append({
                "scenario": self.scenario,
                "seed": self.seed,
                "tick": e.tick,
                "block_id": e.block_id,
                "model_id": e.model_id,
                "layer_idx": e.layer_idx,
                "pool_type": int(e.pool_type),
                "is_hit": e.is_hit,
            })
        return pd.DataFrame(records)

    def to_eviction_dataframe(self) -> pd.DataFrame:
        """Export eviction events as a pandas DataFrame."""
        if not self.eviction_events:
            return pd.DataFrame()

        records = []
        for e in self.eviction_events:
            records.append({
                "scenario": self.scenario,
                "seed": self.seed,
                "tick": e.tick,
                "eviction_id": e.eviction_id,
                "victim_block_id": e.victim_block_id,
                "victim_model_id": e.victim_model_id,
                "victim_layer_idx": e.victim_layer_idx,
                "pool_type": int(e.pool_type),
                "num_candidates": e.num_candidates,
            })
        return pd.DataFrame(records)

    def export_accesses_parquet(self, path: str) -> None:
        """Write access events to a Parquet file."""
        df = self.to_access_dataframe()
        if not df.empty:
            df.to_parquet(path, index=False)

    def export_evictions_parquet(self, path: str) -> None:
        """Write eviction events to a Parquet file."""
        df = self.to_eviction_dataframe()
        if not df.empty:
            df.to_parquet(path, index=False)

    def clear(self) -> None:
        """Reset all collected events."""
        self.access_events.clear()
        self.eviction_events.clear()
        self._next_eviction_id = 0

    @property
    def num_accesses(self) -> int:
        return len(self.access_events)

    @property
    def num_evictions(self) -> int:
        return len(self.eviction_events)

    @property
    def num_hits(self) -> int:
        return sum(1 for e in self.access_events if e.is_hit)

    @property
    def num_misses(self) -> int:
        return sum(1 for e in self.access_events if not e.is_hit)
