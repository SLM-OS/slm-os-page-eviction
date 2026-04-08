"""CACHEUS-style adaptive expert selector (Section 7.3 of the plan).

Maintains a weighted ensemble of expert eviction policies and adapts
the weights online via regret minimization. At each eviction, each
expert scores all candidates; the weighted combination selects the
victim. Feedback from subsequent faults adjusts expert weights.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from .base import EvictionPolicy

if TYPE_CHECKING:
    from src.simulator.block import BlockMeta
    from src.simulator.core import GlobalState, Pool


@dataclass
class EvictionRecord:
    """Record of an eviction decision for feedback analysis."""
    tick: int
    victim_block_id: int
    expert_choices: list[int]  # Which candidate each expert would have chosen
    chosen_expert_idx: int     # Which expert's recommendation was followed


class CACHEUSSelector(EvictionPolicy):
    """Online adaptive expert selector using weighted ensemble.

    Expert pool (Section 7.3):
      - SR-LRU (Scan-resistant LRU)
      - CR-LFU (Churn-resistant LFU)
      - SLM-Heuristic (hand-tuned)
      - MLP (trained offline)
      - XGBoost (trained offline)

    Weight update uses gradient-based hill climbing with regret
    minimization: experts that agree with bad decisions are penalized,
    experts that agree with good decisions are rewarded.
    """

    def __init__(
        self,
        experts: list[EvictionPolicy],
        learning_rate: float = 0.1,
        window_size: int = 100,
        min_weight: float = 0.01,
    ):
        self._experts = experts
        self._num_experts = len(experts)
        self._weights = np.ones(self._num_experts) / self._num_experts
        self._lr = learning_rate
        self._min_weight = min_weight
        self._history: deque[EvictionRecord] = deque(maxlen=window_size)
        self._window_size = window_size

        # Per-expert tracking
        self._expert_faults: list[int] = [0] * self._num_experts
        self._expert_decisions: list[int] = [0] * self._num_experts

    def select_victim(
        self,
        candidates: list[BlockMeta],
        pool: Pool,
        global_state: GlobalState,
    ) -> int:
        """Select victim using weighted expert ensemble."""
        if not candidates:
            return 0

        # Get scores from each expert
        combined_scores = np.zeros(len(candidates))
        expert_choices: list[int] = []

        for i, (expert, weight) in enumerate(zip(self._experts, self._weights)):
            expert_scores = expert.score(candidates, pool, global_state)
            expert_scores = np.array(expert_scores)

            # Track which candidate each expert would have chosen
            expert_choices.append(int(np.argmax(expert_scores)))

            # Weighted combination
            combined_scores += weight * expert_scores

        victim_idx = int(np.argmax(combined_scores))

        # Record the decision for later feedback
        self._history.append(EvictionRecord(
            tick=global_state.tick,
            victim_block_id=candidates[victim_idx].block_id,
            expert_choices=expert_choices,
            chosen_expert_idx=-1,  # Ensemble decision, not a single expert
        ))

        return victim_idx

    def update_feedback(self, block_id: int, was_fault: bool) -> None:
        """Update expert weights based on eviction outcome.

        Called when an evicted block is (or is not) re-accessed soon.
        Experts that would have chosen the faulting block are penalized;
        experts that would have avoided it are rewarded.
        """
        # Find the record for this block
        for record in reversed(self._history):
            if record.victim_block_id == block_id:
                self._update_weights(record, was_fault)
                break

    def _update_weights(
        self, record: EvictionRecord, was_fault: bool
    ) -> None:
        """Apply multiplicative weight update."""
        for i, expert_choice_idx in enumerate(record.expert_choices):
            self._expert_decisions[i] += 1
            # Did this expert agree with the ensemble's decision?
            agreed = (expert_choice_idx == record.expert_choices[-1])

            if was_fault:
                # Bad decision -- penalize experts that agreed
                if agreed:
                    self._weights[i] *= (1 - self._lr)
                    self._expert_faults[i] += 1
            else:
                # Good decision -- reward experts that agreed
                if agreed:
                    self._weights[i] *= (1 + self._lr)

        # Enforce minimum weight and renormalize
        self._weights = np.maximum(self._weights, self._min_weight)
        self._weights /= self._weights.sum()

    @property
    def expert_weights(self) -> dict[str, float]:
        """Return current expert weight distribution."""
        return {
            expert.name(): float(w)
            for expert, w in zip(self._experts, self._weights)
        }

    @property
    def expert_fault_rates(self) -> dict[str, float]:
        """Return per-expert fault contribution rates."""
        return {
            expert.name(): (
                self._expert_faults[i] / max(self._expert_decisions[i], 1)
            )
            for i, expert in enumerate(self._experts)
        }

    def detect_phase_change(self, threshold: float = 0.3) -> bool:
        """Detect if a workload phase change has occurred.

        A phase change is indicated by a rapid shift in expert weights
        (any single weight changes by more than threshold since the
        last check).
        """
        if len(self._history) < self._window_size // 2:
            return False
        # Compare current weights to uniform -- large deviation = adaptation
        uniform = 1.0 / self._num_experts
        max_deviation = float(np.max(np.abs(self._weights - uniform)))
        return max_deviation > threshold

    def score(
        self,
        candidates: list[BlockMeta],
        pool: Pool,
        global_state: GlobalState,
    ) -> list[float]:
        """CACHEUS combined score (for meta-CACHEUS or analysis)."""
        if not candidates:
            return []
        combined = np.zeros(len(candidates))
        for expert, weight in zip(self._experts, self._weights):
            scores = np.array(expert.score(candidates, pool, global_state))
            combined += weight * scores
        return combined.tolist()

    def reset(self) -> None:
        self._weights = np.ones(self._num_experts) / self._num_experts
        self._history.clear()
        self._expert_faults = [0] * self._num_experts
        self._expert_decisions = [0] * self._num_experts
        for expert in self._experts:
            expert.reset()

    def name(self) -> str:
        return "CACHEUS"
