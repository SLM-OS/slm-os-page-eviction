#!/usr/bin/env python3
"""Run the full benchmark suite: all policies x all scenarios x 5 seeds.

Produces comparison tables and summary statistics for the capstone report.

Usage:
    python scripts/benchmark.py [--output-dir data/results]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.features.extractor import FeatureConfig
from src.policies.lru import LRUPolicy
from src.policies.lfu import LFUPolicy
from src.policies.arc import ARCPolicy
from src.policies.slm_heuristic import SLMHeuristicPolicy
from src.policies.belady import BeladyOracle
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
    policies = [
        LRUPolicy(),
        LFUPolicy(),
        ARCPolicy(),
        SLMHeuristicPolicy(),
    ]

    # Load trained ML models if available
    ml_policies = _try_load_ml_policies(model_dir, feature_config)
    policies.extend(ml_policies)

    # Create CACHEUS selector if we have ML experts
    if ml_policies:
        try:
            from src.policies.cacheus import CACHEUSSelector
            experts = [LRUPolicy(), LFUPolicy(), ARCPolicy()] + ml_policies
            cacheus = CACHEUSSelector(experts=experts)
            policies.append(cacheus)
            print(f"  Created CACHEUS selector with {len(experts)} experts")
        except Exception as e:
            print(f"  Warning: failed to create CACHEUS selector: {e}")

    # Generate scenarios
    print("Generating workloads...")
    wg = WorkloadGenerator(seed=42)
    scenarios = {}
    for name in WorkloadGenerator.all_scenario_names():
        scenarios[name] = wg.generate_scenario(name)
        print(f"  {name}: {len(scenarios[name])} requests")

    # Run benchmarks
    print("\nRunning benchmarks...")
    results_df = evaluator.run_benchmark(policies, scenarios, seeds)

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
