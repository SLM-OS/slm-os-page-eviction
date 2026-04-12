#!/usr/bin/env python3
"""Run the full benchmark suite: all policies x all scenarios x 5 seeds.

Produces comparison tables and summary statistics for the capstone report.

Usage:
    python scripts/benchmark.py [--output-dir data/results]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.features.extractor import FeatureConfig
from src.policies.lru import LRUPolicy
from src.policies.lfu import LFUPolicy
from src.policies.arc import ARCPolicy
from src.policies.slm_heuristic import SLMHeuristicPolicy
from src.policies.belady import BeladyOracle
from src.simulator.core import MemoryState, SimController
from src.simulator.trace import TraceCollector
from src.simulator.workload import WorkloadGenerator
from src.training.evaluate import PolicyEvaluator


def _try_load_ml_policies(
    model_dir: Path,
    feature_config: FeatureConfig,
) -> list:
    """Attempt to load trained XGBoost and MLP models. Returns policies found."""
    loaded = []

    xgb_path = model_dir / "xgb_model.json"
    if xgb_path.exists():
        try:
            from src.policies.xgb_policy import XGBPolicy
            from src.training.train_xgb import load_model as load_xgb
            booster = load_xgb(str(xgb_path))
            loaded.append(XGBPolicy(booster, feature_config))
            print(f"  Loaded XGBoost model: {xgb_path}")
        except Exception as e:
            print(f"  Warning: failed to load XGBoost model: {e}")

    mlp_path = model_dir / "mlp_model.pt"
    if mlp_path.exists():
        try:
            from src.policies.mlp_policy import MLPPolicy
            from src.training.train_mlp import load_model as load_mlp
            mlp = load_mlp(str(mlp_path), feature_config.num_features)
            loaded.append(MLPPolicy(mlp, feature_config))
            print(f"  Loaded MLP model: {mlp_path}")
        except Exception as e:
            print(f"  Warning: failed to load MLP model: {e}")

    return loaded


def main() -> None:
    parser = argparse.ArgumentParser(description="Run benchmark suite")
    parser.add_argument(
        "--output-dir", default="data/results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--model-dir", default="data/models",
        help="Directory containing trained models",
    )
    parser.add_argument(
        "--seeds", type=int, default=5,
        help="Number of random seeds",
    )
    parser.add_argument(
        "--no-predicted-reuse", action="store_true",
        help="Use 26-feature models (no predicted_reuse_dist)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(args.model_dir)

    feature_config = FeatureConfig(
        use_predicted_reuse=not args.no_predicted_reuse
    )

    seeds = [42, 123, 456, 789, 1337][:args.seeds]
    evaluator = PolicyEvaluator(weight_blocks=64, workspace_blocks=32)

    # Classical policies (always available)
    classical_policies = [
        LRUPolicy(),
        LFUPolicy(),
        ARCPolicy(),
        SLMHeuristicPolicy(),
    ]

    # Load trained ML models if available
    ml_policies = _try_load_ml_policies(model_dir, feature_config)

    # Create CACHEUS selector if we have ML experts
    cacheus_policy = None
    if ml_policies:
        try:
            from src.policies.cacheus import CACHEUSSelector
            experts = [LRUPolicy(), LFUPolicy(), ARCPolicy()] + ml_policies
            cacheus_policy = CACHEUSSelector(experts=experts)
            print(f"  Created CACHEUS selector with {len(experts)} experts")
        except Exception as e:
            print(f"  Warning: failed to create CACHEUS selector: {e}")

    scenario_names = WorkloadGenerator.all_scenario_names()

    # Combine all non-oracle policies
    policies = classical_policies + ml_policies
    if cacheus_policy:
        policies.append(cacheus_policy)

    # Run benchmarks with progress — regenerate workloads per seed
    total_runs = (len(policies) + 1) * len(scenario_names) * len(seeds)
    print(f"\nRunning benchmarks ({len(policies)+1} policies x "
          f"{len(scenario_names)} scenarios x {len(seeds)} seeds = {total_runs} runs)...")
    all_results = []
    run_count = 0

    for scenario_name in scenario_names:
        for seed in seeds:
            # Generate workload for this seed
            wg = WorkloadGenerator(seed=seed)
            requests = wg.generate_scenario(scenario_name)

            # Build Belady oracle for this specific workload
            memory = MemoryState(evaluator.weight_blocks, evaluator.workspace_blocks)
            trace = TraceCollector(scenario=scenario_name, seed=seed)
            lru_sim = SimController(
                memory=memory, policy=LRUPolicy(), trace_collector=trace
            )
            lru_sim.run(requests)
            oracle = BeladyOracle(future_accesses=trace.access_events)

            # Run Belady
            oracle.reset()
            result = evaluator.run_policy(oracle, requests, scenario_name, seed)
            all_results.append(result)
            run_count += 1
            print(f"  [{run_count}/{total_runs}] Belady-Optimal / {scenario_name} / {seed}: "
                  f"faults={result.metrics.total_faults}")

            # Run all other policies
            for policy in policies:
                policy.reset()
                result = evaluator.run_policy(
                    policy, requests, scenario_name, seed
                )
                all_results.append(result)
                run_count += 1
                print(f"  [{run_count}/{total_runs}] {policy.name()} / {scenario_name} / {seed}: "
                      f"faults={result.metrics.total_faults}")

    # Build DataFrame with normalized fault rates
    results_rows = []
    # Index results for baseline lookup
    result_index = {}
    for r in all_results:
        result_index[(r.policy_name, r.scenario, r.seed)] = r

    from src.simulator.metrics import compute_normalized_fault_rate

    for r in all_results:
        lru_key = ("LRU", r.scenario, r.seed)
        belady_key = ("Belady-Optimal", r.scenario, r.seed)
        lru_faults = result_index[lru_key].metrics.total_faults
        optimal_faults = result_index[belady_key].metrics.total_faults
        norm_rate = compute_normalized_fault_rate(
            r.metrics.total_faults, optimal_faults, lru_faults
        )
        results_rows.append({
            "policy": r.policy_name,
            "scenario": r.scenario,
            "seed": r.seed,
            "total_accesses": r.metrics.total_accesses,
            "total_faults": r.metrics.total_faults,
            "fault_rate": r.metrics.fault_rate,
            "normalized_fault_rate": norm_rate,
            "total_evictions": r.metrics.total_evictions,
        })

    results_df = pd.DataFrame(results_rows)

    # Save results
    results_file = output_dir / "benchmark_results.csv"
    results_df.to_csv(results_file, index=False)
    print(f"\nResults saved: {results_file}")

    # Summary
    summary = evaluator.summarize(results_df)
    print("\n--- Summary ---")
    print(summary.to_string(index=False))

    summary_file = output_dir / "benchmark_summary.csv"
    summary.to_csv(summary_file, index=False)


if __name__ == "__main__":
    main()
