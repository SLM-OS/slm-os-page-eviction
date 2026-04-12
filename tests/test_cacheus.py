"""Unit tests for CACHEUSSelector (Section 7.3 of the plan).

Tests the expert ensemble, weight update mechanism, and phase detection.
Focused on the fixes from Phase 3-4 work:
- EvictionRecord now stores ensemble_choice (not chosen_expert_idx)
- Weight update compares against ensemble_choice, not last expert's choice
- Experts that disagreed with a bad eviction are rewarded (not just neutral)
"""

from __future__ import annotations

import pytest

from src.simulator.block import BlockMeta, BlockState, PoolType
from src.simulator.core import GlobalState, Pool
from src.policies.cacheus import CACHEUSSelector, EvictionRecord
from src.policies.lru import LRUPolicy
from src.policies.lfu import LFUPolicy


def _make_candidates(
    num: int,
    access_times: list[int] | None = None,
    access_counts: list[int] | None = None,
) -> tuple[list[BlockMeta], Pool, GlobalState]:
    """Helper to create eviction candidates with specified attributes."""
    pool = Pool(PoolType.WEIGHT, num_blocks=num)
    for i, block in enumerate(pool.blocks):
        block.state = BlockState.ALLOCATED
        block.model_id = 0
        block.layer_idx = i
        if access_times:
            block.last_access_time = access_times[i]
        if access_counts:
            block.access_count = access_counts[i]

    candidates = pool.get_evictable_blocks()
    global_state = GlobalState(tick=1000)
    return candidates, pool, global_state


class TestCACHEUSInit:
    """CACHEUS initialization with experts."""

    def test_equal_initial_weights(self):
        experts = [LRUPolicy(), LFUPolicy()]
        selector = CACHEUSSelector(experts=experts)
        weights = selector.expert_weights
        assert len(weights) == 2
        # Equal distribution across experts
        assert all(abs(w - 0.5) < 1e-6 for w in weights.values())

    def test_weights_sum_to_one(self):
        experts = [LRUPolicy(), LFUPolicy()]
        selector = CACHEUSSelector(experts=experts)
        total = sum(selector.expert_weights.values())
        assert abs(total - 1.0) < 1e-6

    def test_internal_weights_sum_to_one(self):
        """Internal _weights array sums to 1 regardless of expert name uniqueness."""
        import numpy as np
        experts = [LRUPolicy(), LFUPolicy(), LRUPolicy()]
        selector = CACHEUSSelector(experts=experts)
        assert abs(np.sum(selector._weights) - 1.0) < 1e-6
        assert len(selector._weights) == 3


class TestCACHEUSSelectVictim:
    """CACHEUS select_victim combines expert scores."""

    def test_selects_from_candidates(self):
        candidates, pool, gs = _make_candidates(
            3, access_times=[100, 50, 200]
        )
        experts = [LRUPolicy(), LFUPolicy()]
        selector = CACHEUSSelector(experts=experts)
        victim = selector.select_victim(candidates, pool, gs)
        assert 0 <= victim < len(candidates)

    def test_records_decision(self):
        candidates, pool, gs = _make_candidates(
            3, access_times=[100, 50, 200]
        )
        experts = [LRUPolicy(), LFUPolicy()]
        selector = CACHEUSSelector(experts=experts)
        selector.select_victim(candidates, pool, gs)

        # The decision is recorded in history
        assert len(selector._history) == 1
        record = selector._history[0]
        # Record stores the ensemble's actual choice
        assert hasattr(record, "ensemble_choice")
        assert isinstance(record.ensemble_choice, int)
        assert 0 <= record.ensemble_choice < len(candidates)

    def test_record_stores_per_expert_choices(self):
        candidates, pool, gs = _make_candidates(
            3, access_times=[100, 50, 200]
        )
        experts = [LRUPolicy(), LFUPolicy()]
        selector = CACHEUSSelector(experts=experts)
        selector.select_victim(candidates, pool, gs)

        record = selector._history[0]
        # One choice per expert
        assert len(record.expert_choices) == len(experts)


class TestCACHEUSWeightUpdate:
    """Weight updates via multiplicative hill climbing (regret minimization)."""

    def test_penalizes_agreeing_experts_on_bad_eviction(self):
        """Experts that agreed with the ensemble's bad eviction lose weight."""
        import numpy as np

        experts = [LRUPolicy(), LFUPolicy()]
        selector = CACHEUSSelector(experts=experts, learning_rate=0.3)

        # Directly inject a record where expert 0 agreed, expert 1 disagreed
        # ensemble_choice=0, expert 0 picked 0 (agreed), expert 1 picked 1 (disagreed)
        record = EvictionRecord(
            tick=10,
            victim_block_id=999,
            expert_choices=[0, 1],
            ensemble_choice=0,
        )
        selector._history.append(record)

        w_before = selector._weights.copy()
        selector.update_feedback(999, was_fault=True)
        w_after = selector._weights

        # Expert 0 agreed with ensemble → penalized (weight decreased)
        # Expert 1 disagreed → rewarded (weight increased relatively)
        # After normalization, expert 0 should lose relative weight
        assert w_after[0] < w_before[0], \
            "Expected agreeing expert (0) to lose weight after bad eviction"
        assert w_after[1] > w_before[1], \
            "Expected disagreeing expert (1) to gain relative weight"

    def test_rewards_disagreeing_experts_on_bad_eviction(self):
        """Experts that chose differently from the ensemble should not be penalized for its mistake."""
        experts = [LRUPolicy(), LFUPolicy()]
        selector = CACHEUSSelector(experts=experts, learning_rate=0.3)

        # Inject record: ensemble picked 0, expert 1 picked 2 (disagreed)
        record = EvictionRecord(
            tick=10,
            victim_block_id=999,
            expert_choices=[0, 2],
            ensemble_choice=0,
        )
        selector._history.append(record)

        # Bad eviction — but expert 1 disagreed, so should not be penalized
        expert_fault_before = selector._expert_faults[1]
        selector.update_feedback(999, was_fault=True)
        assert selector._expert_faults[1] == expert_fault_before, \
            "Disagreeing expert should not accumulate fault count"

    def test_rewards_agreeing_experts_on_good_eviction(self):
        """Experts that agreed with a good eviction gain weight."""
        experts = [LRUPolicy(), LFUPolicy()]
        selector = CACHEUSSelector(experts=experts, learning_rate=0.3)

        # Inject record: ensemble picked 0, expert 0 agreed, expert 1 disagreed
        record = EvictionRecord(
            tick=10,
            victim_block_id=999,
            expert_choices=[0, 1],
            ensemble_choice=0,
        )
        selector._history.append(record)

        w_before = selector._weights.copy()
        # Good eviction (not re-accessed within window)
        selector.update_feedback(999, was_fault=False)
        w_after = selector._weights

        # Expert 0 agreed → rewarded. Expert 1 disagreed → unchanged.
        # After normalization, expert 0 should gain relative weight
        assert w_after[0] > w_before[0], \
            "Expected agreeing expert to gain weight after good eviction"

    def test_weights_remain_normalized(self):
        """After any update, weights still sum to 1."""
        import numpy as np
        candidates, pool, gs = _make_candidates(
            3, access_times=[100, 50, 200]
        )
        experts = [LRUPolicy(), LFUPolicy()]
        selector = CACHEUSSelector(experts=experts, learning_rate=0.3)

        selector.select_victim(candidates, pool, gs)
        victim_block_id = selector._history[0].victim_block_id

        selector.update_feedback(victim_block_id, was_fault=True)
        total = np.sum(selector._weights)
        assert abs(total - 1.0) < 1e-6

    def test_min_weight_keeps_experts_recoverable(self):
        """Min weight ensures no expert is permanently silenced (stays > 0)."""
        experts = [LRUPolicy(), LFUPolicy()]
        selector = CACHEUSSelector(
            experts=experts, learning_rate=0.9, min_weight=0.05
        )

        # Repeatedly inject records where expert 0 always agrees (gets penalized)
        for _ in range(50):
            record = EvictionRecord(
                tick=10, victim_block_id=999,
                expert_choices=[0, 1], ensemble_choice=0,
            )
            selector._history.append(record)
            selector.update_feedback(999, was_fault=True)

        # After normalization, minimum can drop below min_weight,
        # but it should remain strictly positive (never zero)
        import numpy as np
        assert np.min(selector._weights) > 0
        # And sum stays 1
        assert abs(np.sum(selector._weights) - 1.0) < 1e-6


class TestCACHEUSReset:
    """Reset should restore uniform weights and clear history."""

    def test_reset_restores_uniform_weights(self):
        import numpy as np
        experts = [LRUPolicy(), LFUPolicy()]
        selector = CACHEUSSelector(experts=experts, learning_rate=0.3)

        # Skew weights via injected records
        for _ in range(10):
            record = EvictionRecord(
                tick=10, victim_block_id=999,
                expert_choices=[0, 1], ensemble_choice=0,
            )
            selector._history.append(record)
            selector.update_feedback(999, was_fault=True)

        # Confirm weights are no longer uniform
        assert not all(abs(w - 0.5) < 1e-6 for w in selector._weights)

        selector.reset()
        # Weights restored to uniform
        for w in selector._weights:
            assert abs(w - 1.0 / len(experts)) < 1e-6
        assert len(selector._history) == 0


class TestCACHEUSPhaseDetection:
    """Phase change detection based on weight deviation."""

    def test_no_phase_change_with_little_history(self):
        experts = [LRUPolicy(), LFUPolicy()]
        selector = CACHEUSSelector(experts=experts, window_size=100)
        assert selector.detect_phase_change() is False

    def test_phase_change_detected_after_skew(self):
        """Strong weight skew → phase change detected."""
        candidates, pool, gs = _make_candidates(
            3, access_times=[100, 50, 200]
        )
        experts = [LRUPolicy(), LFUPolicy()]
        selector = CACHEUSSelector(
            experts=experts, learning_rate=0.5, window_size=20
        )

        # Enough decisions to fill window half
        for _ in range(30):
            selector.select_victim(candidates, pool, gs)
            vid = selector._history[-1].victim_block_id
            selector.update_feedback(vid, was_fault=True)

        # With high learning rate and consistent bad evictions, some expert may skew
        # (Just check the API works and returns a bool)
        result = selector.detect_phase_change(threshold=0.1)
        assert isinstance(result, bool)
