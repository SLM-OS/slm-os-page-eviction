"""Policy evaluation: runs policies in the simulator and compares metrics.

Implements the benchmark suite from Section 7.4 and Phase 7. Runs each
policy on each scenario, collects metrics, and produces comparison tables.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.simulator.core import MemoryState, SimController
from src.simulator.metrics import (
    PolicyMetrics,
    compute_fault_rate,
    compute_normalized_fault_rate,
)
from src.simulator.trace import TraceCollector

if TYPE_CHECKING:
    from src.policies.base import EvictionPolicy
    from src.simulator.workload import AccessRequest


@dataclass
class BenchmarkResult:
    """Results from running a policy on a scenario."""
    policy_name: str
    scenario: str
    seed: int
    metrics: PolicyMetrics
    trace: TraceCollector


class PolicyEvaluator:
    """Runs eviction policies in the simulator and collects metrics.

    Supports running multiple policies across multiple scenarios with
    multiple seeds for statistical comparison.
    """

    def __init__(
        self,
        weight_blocks: int = 64,
        workspace_blocks: int = 32,
    ):
        self.weight_blocks = weight_blocks
        self.workspace_blocks = workspace_blocks

    def run_policy(
        self,
        policy: EvictionPolicy,
        requests: list[AccessRequest],
        scenario: str = "",
        seed: int = 0,
    ) -> BenchmarkResult:
        """Run a single policy on a sequence of access requests.

        Args:
            policy: The eviction policy to evaluate.
            requests: Sequence of memory access requests.
            scenario: Scenario name for labeling.
            seed: Random seed for labeling.

        Returns:
            BenchmarkResult with metrics and trace data.
        """
        memory = MemoryState(self.weight_blocks, self.workspace_blocks)
        trace = TraceCollector(scenario=scenario, seed=seed)
        sim = SimController(memory=memory, policy=policy, trace_collector=trace)

        policy.reset()
        sim.run(requests)

        metrics = PolicyMetrics(
            policy_name=policy.name(),
            scenario=scenario,
            seed=seed,
            total_accesses=sim.total_accesses,
            total_faults=sim.total_faults,
            total_evictions=sim.total_evictions,
            total_ticks=sim.tick,
        )

        return BenchmarkResult(
            policy_name=policy.name(),
            scenario=scenario,
            seed=seed,
            metrics=metrics,
            trace=trace,
        )

    def run_benchmark(
        self,
        policies: list[EvictionPolicy],
        scenarios: dict[str, list[AccessRequest]],
        seeds: list[int] | None = None,
    ) -> pd.DataFrame:
        """Run all policies on all scenarios and produce a comparison table.

        Args:
            policies: List of eviction policies to compare.
            scenarios: Dict mapping scenario name to access requests.
            seeds: Random seeds for repeated runs.

        Returns:
            DataFrame with columns: policy, scenario, seed, fault_rate,
            normalized_fault_rate, eviction_cost, etc.
        """
        if seeds is None:
            seeds = [42]

        results: list[dict] = []

        # First pass: collect all results and find Belady/LRU baselines
        all_benchmarks: dict[tuple[str, str, int], BenchmarkResult] = {}

        for scenario_name, requests in scenarios.items():
            for seed in seeds:
                for policy in policies:
                    result = self.run_policy(
                        policy, requests, scenario_name, seed
                    )
                    key = (policy.name(), scenario_name, seed)
                    all_benchmarks[key] = result

        # Second pass: compute normalized fault rates
        for (policy_name, scenario_name, seed), result in all_benchmarks.items():
            lru_key = ("LRU", scenario_name, seed)
            belady_key = ("Belady-Optimal", scenario_name, seed)

            lru_faults = (
                all_benchmarks[lru_key].metrics.total_faults
                if lru_key in all_benchmarks else result.metrics.total_faults
            )
            optimal_faults = (
                all_benchmarks[belady_key].metrics.total_faults
                if belady_key in all_benchmarks else 0
            )

            norm_rate = compute_normalized_fault_rate(
                result.metrics.total_faults, optimal_faults, lru_faults
            )

            results.append({
                "policy": policy_name,
                "scenario": scenario_name,
                "seed": seed,
                "total_accesses": result.metrics.total_accesses,
                "total_faults": result.metrics.total_faults,
                "fault_rate": result.metrics.fault_rate,
                "normalized_fault_rate": norm_rate,
                "total_evictions": result.metrics.total_evictions,
            })

        return pd.DataFrame(results)

    def summarize(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """Summarize benchmark results: mean and std across seeds.

        Returns a pivot table with policies as rows, scenarios as columns,
        and mean fault rate as values.
        """
        summary = results_df.groupby(["policy", "scenario"]).agg(
            fault_rate_mean=("fault_rate", "mean"),
            fault_rate_std=("fault_rate", "std"),
            norm_rate_mean=("normalized_fault_rate", "mean"),
            norm_rate_std=("normalized_fault_rate", "std"),
        ).reset_index()

        return summary
