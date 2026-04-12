"""Unit tests for analysis helpers in scripts/analyze_benchmark.py.

The benchmark CSV has a fixed schema; these tests verify the pivot, t-test,
and failure-analysis helpers handle the edge cases (zero variance, missing
policies, etc.) that arise in practice.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _make_benchmark_df(
    policies: list[str] | None = None,
    scenarios: list[str] | None = None,
    seeds: list[int] | None = None,
    fault_rates: dict[tuple[str, str], list[float]] | None = None,
) -> pd.DataFrame:
    """Construct a synthetic benchmark DataFrame for testing."""
    if policies is None:
        policies = ["LRU", "XGBoost", "MLP", "Belady-Optimal"]
    if scenarios is None:
        scenarios = ["scenario_a", "scenario_b"]
    if seeds is None:
        seeds = [42, 123, 456]
    if fault_rates is None:
        fault_rates = {}

    rows = []
    for policy in policies:
        for scenario in scenarios:
            rates = fault_rates.get((policy, scenario), [0.5] * len(seeds))
            for seed, rate in zip(seeds, rates):
                rows.append({
                    "policy": policy,
                    "scenario": scenario,
                    "seed": seed,
                    "total_accesses": 1000,
                    "total_faults": int(rate * 1000),
                    "fault_rate": rate,
                    "normalized_fault_rate": rate,
                    "total_evictions": int(rate * 800),
                })
    return pd.DataFrame(rows)


class TestPivotResults:
    """Pivot table construction."""

    def test_pivot_shape(self):
        from scripts.analyze_benchmark import pivot_results
        df = _make_benchmark_df()
        pivot = pivot_results(df)
        # Default 4 policies × 2 scenarios; some not in canonical orderings
        # (only Belady-Optimal, XGBoost, MLP, LRU appear in the canonical order)
        assert "scenario_a" not in pivot.columns  # not in canonical scenario list
        # But Belady, XGBoost, MLP, LRU are in canonical policy order
        assert set(pivot.index) <= {"Belady-Optimal", "XGBoost", "MLP", "LRU"}

    def test_pivot_canonical_scenarios(self):
        from scripts.analyze_benchmark import pivot_results
        df = _make_benchmark_df(
            scenarios=["single_inference", "multi_model"],
        )
        pivot = pivot_results(df)
        assert list(pivot.columns) == ["single_inference", "multi_model"]

    def test_pivot_values(self):
        from scripts.analyze_benchmark import pivot_results
        df = _make_benchmark_df(
            policies=["LRU", "XGBoost"],
            scenarios=["single_inference"],
            fault_rates={
                ("LRU", "single_inference"): [1.0, 1.0, 1.0],
                ("XGBoost", "single_inference"): [0.2, 0.3, 0.4],
            },
        )
        pivot = pivot_results(df)
        assert abs(pivot.loc["LRU", "single_inference"] - 1.0) < 1e-9
        assert abs(pivot.loc["XGBoost", "single_inference"] - 0.3) < 1e-9


class TestPairedTTest:
    """Per-scenario paired t-tests."""

    def test_zero_variance_handled_gracefully(self):
        """When seeds produce identical results, p=1.0 is returned (not NaN)."""
        from scripts.analyze_benchmark import paired_ttest_pairs
        df = _make_benchmark_df(
            policies=["LRU", "XGBoost"],
            scenarios=["s1"],
            fault_rates={
                ("LRU", "s1"): [1.0, 1.0, 1.0],
                ("XGBoost", "s1"): [0.0, 0.0, 0.0],
            },
        )
        results = paired_ttest_pairs(df, "LRU", "XGBoost")
        r = results["s1"]
        assert r["valid"] is True
        assert r["p"] == 1.0
        assert r["significant"] is False
        assert "note" in r

    def test_significant_difference_flagged(self):
        from scripts.analyze_benchmark import paired_ttest_pairs
        df = _make_benchmark_df(
            policies=["LRU", "XGBoost"],
            scenarios=["s1"],
            fault_rates={
                ("LRU", "s1"): [0.9, 0.92, 0.88],
                ("XGBoost", "s1"): [0.2, 0.25, 0.18],
            },
        )
        results = paired_ttest_pairs(df, "LRU", "XGBoost")
        r = results["s1"]
        assert r["valid"] is True
        assert r["significant"] is True
        assert r["p"] < 0.05

    def test_insufficient_seeds_invalid(self):
        from scripts.analyze_benchmark import paired_ttest_pairs
        df = _make_benchmark_df(
            policies=["LRU", "XGBoost"],
            scenarios=["s1"],
            seeds=[42],
            fault_rates={
                ("LRU", "s1"): [1.0],
                ("XGBoost", "s1"): [0.0],
            },
        )
        results = paired_ttest_pairs(df, "LRU", "XGBoost")
        assert results["s1"]["valid"] is False


class TestFailureAnalysis:
    """Identifies scenarios where each policy underperforms a threshold."""

    def test_no_failures_under_threshold(self):
        from scripts.analyze_benchmark import failure_analysis
        df = _make_benchmark_df(
            policies=["XGBoost"],
            scenarios=["s1"],
            fault_rates={("XGBoost", "s1"): [0.1, 0.2, 0.15]},
        )
        failures = failure_analysis(df, threshold=0.5)
        assert len(failures) == 0

    def test_failures_above_threshold(self):
        from scripts.analyze_benchmark import failure_analysis
        df = _make_benchmark_df(
            policies=["LRU"],
            scenarios=["s1", "s2"],
            fault_rates={
                ("LRU", "s1"): [0.9, 0.95, 0.92],
                ("LRU", "s2"): [0.1, 0.15, 0.12],
            },
        )
        failures = failure_analysis(df, threshold=0.5)
        assert len(failures) == 1
        assert failures.iloc[0]["scenario"] == "s1"


class TestAllPairwiseTests:
    """All-pairs t-test wrapper produces well-formed results."""

    def test_pairs_count_correct(self):
        from scripts.analyze_benchmark import all_pairwise_tests
        df = _make_benchmark_df(
            policies=["A", "B", "C"],
            scenarios=["s1"],
            fault_rates={
                ("A", "s1"): [0.5, 0.55, 0.5],
                ("B", "s1"): [0.3, 0.35, 0.32],
                ("C", "s1"): [0.7, 0.72, 0.68],
            },
        )
        # 3 policies → 3 pairs (AB, AC, BC), 1 scenario each = 3 rows
        result = all_pairwise_tests(df, ["A", "B", "C"])
        assert len(result) == 3
        assert set(result.columns) >= {
            "policy_a", "policy_b", "scenario", "t", "p_value", "significant",
        }


class TestCacheusTuningHelpers:
    """Tests for helper functions in scripts/tune_cacheus.py."""

    def test_build_expert_pool_classical_only(self):
        from scripts.tune_cacheus import build_expert_pool
        from src.policies.lru import LRUPolicy
        from src.policies.lfu import LFUPolicy

        pool = build_expert_pool("classical_only", ml_experts=[])
        assert len(pool) == 4
        names = [e.name() for e in pool]
        assert "LRU" in names
        assert "LFU" in names
        assert "ARC" in names
        assert "SLM-Heuristic" in names

    def test_build_expert_pool_includes_ml(self):
        from scripts.tune_cacheus import build_expert_pool

        # Mock ML experts: any object with a name() method
        class FakeExpert:
            def __init__(self, name):
                self._name = name
            def name(self):
                return self._name

        ml = [FakeExpert("XGBoost"), FakeExpert("MLP")]

        pool_all5 = build_expert_pool("all_5", ml_experts=ml)
        assert len(pool_all5) == 5
        assert "XGBoost" in [e.name() for e in pool_all5]

        pool_ml_only = build_expert_pool("ml_only", ml_experts=ml)
        assert len(pool_ml_only) == 2

        pool_ml_lru = build_expert_pool("ml_plus_lru", ml_experts=ml)
        assert len(pool_ml_lru) == 3
        assert "LRU" in [e.name() for e in pool_ml_lru]

    def test_build_expert_pool_unknown_raises(self):
        from scripts.tune_cacheus import build_expert_pool

        with pytest.raises(KeyError):
            build_expert_pool("nonexistent_pool", ml_experts=[])

    def test_build_belady_oracle_from_requests(self):
        """build_belady_oracle runs LRU first then constructs an oracle."""
        from scripts.tune_cacheus import build_belady_oracle
        from src.policies.belady import BeladyOracle
        from src.simulator.workload import WorkloadGenerator

        wg = WorkloadGenerator(seed=42)
        requests = wg.generate_scenario("burst_load")
        oracle = build_belady_oracle(requests, weight_blocks=8, workspace_blocks=4)
        assert isinstance(oracle, BeladyOracle)
        # Oracle should have access events from the LRU pre-run
        assert len(oracle._future_accesses) > 0


class TestTrajectoryPlotting:
    """Tests for adaptation speed analysis in scripts/plot_trajectories.py."""

    def test_adaptation_speed_returns_none_for_empty(self):
        from scripts.plot_trajectories import adaptation_speed
        traj = {"ticks": [], "weights": []}
        assert adaptation_speed(traj) is None

    def test_adaptation_speed_returns_none_for_single_decision(self):
        from scripts.plot_trajectories import adaptation_speed
        import numpy as np
        traj = {"ticks": [10], "weights": np.array([[0.5, 0.5]])}
        assert adaptation_speed(traj) is None

    def test_adaptation_speed_detects_convergence(self):
        """Once weights stabilize, return the tick of first stable difference."""
        from scripts.plot_trajectories import adaptation_speed
        import numpy as np

        # Weights change initially, then stabilize at tick 30
        weights = np.array([
            [0.5, 0.5],
            [0.3, 0.7],
            [0.2, 0.8],   # diff: 0.2 > threshold
            [0.2, 0.8],   # diff: 0.0 < 0.05 → converged at tick 40
            [0.2, 0.8],
        ])
        traj = {"ticks": [10, 20, 30, 40, 50], "weights": weights}
        result = adaptation_speed(traj, threshold=0.05)
        # Convergence is reported at the tick of the second weight in the converged pair.
        # diff[i] is between weights[i+1] and weights[i]; converged when diff < threshold.
        # diff = [0.4, 0.2, 0.0, 0.0] → first below threshold is index 2 → ticks[2] = 30
        assert result == 30

    def test_adaptation_speed_no_convergence(self):
        """If weights never stabilize, return None."""
        from scripts.plot_trajectories import adaptation_speed
        import numpy as np

        # Weights oscillate
        weights = np.array([
            [0.5, 0.5],
            [0.2, 0.8],
            [0.7, 0.3],
            [0.1, 0.9],
        ])
        traj = {"ticks": [10, 20, 30, 40], "weights": weights}
        assert adaptation_speed(traj, threshold=0.01) is None


class TestCacheusTrajectorySerialization:
    """Regression test: numpy int64 ticks must serialize to JSON.

    The trajectory recording in CACHEUSSelector stores `(tick, weights)`
    tuples where `tick` comes from GlobalState.tick — which is int but may
    be wrapped in numpy types when set from numpy operations. The full
    1h+ tune_cacheus run failed at the very last step because json.dump
    can't serialize np.int64. This test exercises the serialization shape
    used in tune_cacheus.main().
    """

    def test_int64_ticks_serialize(self, tmp_path):
        import json
        import numpy as np

        # Mimic the structure tune_cacheus.main() builds before json.dump
        trajectories = {
            "scenario_a": {
                "expert_names": ["XGBoost", "MLP"],
                "ticks": [np.int64(10), np.int64(20), np.int64(30)],
                "weights": np.array([[0.5, 0.5], [0.4, 0.6], [0.3, 0.7]]),
                "phase_changes": [np.int64(20)],
            }
        }

        # The serialization block from tune_cacheus
        traj_serializable = {
            name: {
                "expert_names": list(traj.get("expert_names", [])),
                "ticks": [int(t) for t in traj["ticks"]],
                "weights": (
                    traj["weights"].tolist()
                    if hasattr(traj["weights"], "tolist") else list(traj["weights"])
                ),
                "phase_changes": [int(t) for t in traj.get("phase_changes", [])],
            }
            for name, traj in trajectories.items()
        }

        out = tmp_path / "trajectories.json"
        with open(out, "w") as f:
            json.dump(traj_serializable, f)

        # Round-trip and verify
        with open(out) as f:
            loaded = json.load(f)
        assert loaded["scenario_a"]["ticks"] == [10, 20, 30]
        assert loaded["scenario_a"]["phase_changes"] == [20]
        assert loaded["scenario_a"]["weights"][0] == [0.5, 0.5]
