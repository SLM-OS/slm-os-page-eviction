"""Policy evaluation metrics (Section 5.2 of the plan).

Computes fault rate, normalized fault rate, eviction cost, throughput,
and deadline miss rate for comparing eviction policies.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PolicyMetrics:
    """Aggregated metrics from a simulation run."""
    policy_name: str
    scenario: str
    seed: int
    total_accesses: int = 0
    total_faults: int = 0
    total_evictions: int = 0
    total_eviction_cost: float = 0.0
    inferences_completed: int = 0
    total_ticks: int = 0
    deadline_tasks: int = 0
    deadline_misses: int = 0

    @property
    def fault_rate(self) -> float:
        """Faults / total accesses. Lower is better."""
        if self.total_accesses == 0:
            return 0.0
        return self.total_faults / self.total_accesses

    @property
    def throughput(self) -> float:
        """Inferences completed / total ticks. Higher is better."""
        if self.total_ticks == 0:
            return 0.0
        return self.inferences_completed / self.total_ticks

    @property
    def deadline_miss_rate(self) -> float:
        """Deadline misses / total deadline tasks. 0.0 = no misses."""
        if self.deadline_tasks == 0:
            return 0.0
        return self.deadline_misses / self.deadline_tasks

    @property
    def avg_eviction_cost(self) -> float:
        """Average cost per eviction (accounts for GPU remap + writeback)."""
        if self.total_evictions == 0:
            return 0.0
        return self.total_eviction_cost / self.total_evictions


def compute_fault_rate(faults: int, accesses: int) -> float:
    """Compute raw fault rate."""
    if accesses == 0:
        return 0.0
    return faults / accesses


def compute_normalized_fault_rate(
    policy_faults: int,
    optimal_faults: int,
    lru_faults: int,
) -> float:
    """Compute normalized fault rate (Section 5.2).

    Returns a value where 0.0 = optimal (Belady) and 1.0 = LRU baseline.
    Values below 0.0 are impossible; values above 1.0 mean worse than LRU.
    """
    denominator = lru_faults - optimal_faults
    if denominator == 0:
        # LRU is already optimal (no room for improvement)
        return 0.0
    return (policy_faults - optimal_faults) / denominator


def compute_eviction_cost(
    is_dirty: bool,
    is_gpu_mapped: bool,
    base_cost: float = 1.0,
    writeback_cost: float = 0.3,
    gpu_unmap_cost: float = 0.5,
) -> float:
    """Compute the cost of evicting a single block.

    Accounts for writeback overhead (dirty blocks) and GPU unmapping cost.
    """
    cost = base_cost
    if is_dirty:
        cost += writeback_cost
    if is_gpu_mapped:
        cost += gpu_unmap_cost
    return cost
