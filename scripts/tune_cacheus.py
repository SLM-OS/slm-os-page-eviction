#!/usr/bin/env python3
"""CACHEUS hyperparameter tuning and online evaluation (Phase 5).

Sweeps learning rate × window size × expert pool composition; runs each
config across all scenarios; reports the best config and per-scenario
weight trajectories.

Usage:
    python scripts/tune_cacheus.py --model-dir data/models/ --output-dir data/results/
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.extractor import FeatureConfig
from src.policies.arc import ARCPolicy
from src.policies.belady import BeladyOracle
from src.policies.cacheus import CACHEUSSelector
from src.policies.lfu import LFUPolicy
from src.policies.lru import LRUPolicy
from src.policies.slm_heuristic import SLMHeuristicPolicy
from src.simulator.core import MemoryState, SimController
from src.simulator.metrics import compute_normalized_fault_rate
from src.simulator.trace import TraceCollector
from src.simulator.workload import WorkloadGenerator
from src.training.evaluate import PolicyEvaluator


def load_ml_experts(model_dir: Path, feature_config: FeatureConfig) -> list:
    """Load XGBoost and MLP policies from disk."""
    experts = []
    xgb_path = model_dir / "xgb_model.json"
    if xgb_path.exists():
        from src.policies.xgb_policy import XGBPolicy
        from src.training.train_xgb import load_model as load_xgb
        experts.append(XGBPolicy(load_xgb(str(xgb_path)), feature_config))

    mlp_path = model_dir / "mlp_model.pt"
    if mlp_path.exists():
        from src.policies.mlp_policy import MLPPolicy
        from src.training.train_mlp import load_model as load_mlp
        experts.append(MLPPolicy(
            load_mlp(str(mlp_path), feature_config.num_features),
            feature_config,
        ))
    return experts


def build_expert_pool(name: str, ml_experts: list) -> list:
    """Return an ordered expert list for the named pool composition."""
    pools = {
        "all_5": [LRUPolicy(), LFUPolicy(), ARCPolicy()] + ml_experts,
        "ml_only": list(ml_experts),
        "ml_plus_lru": [LRUPolicy()] + list(ml_experts),
        "classical_only": [LRUPolicy(), LFUPolicy(), ARCPolicy(), SLMHeuristicPolicy()],
    }
    return pools[name]


def build_belady_oracle(scenario_requests, weight_blocks=64, workspace_blocks=32):
    """Build a Belady oracle for normalization baseline."""
    memory = MemoryState(weight_blocks, workspace_blocks)
    trace = TraceCollector()
    sim = SimController(memory=memory, policy=LRUPolicy(), trace_collector=trace)
    sim.run(scenario_requests)
    return BeladyOracle(future_accesses=trace.access_events)


def evaluate_cacheus(
    cacheus: CACHEUSSelector,
    scenario_requests,
    scenario_name: str,
    seed: int,
    evaluator: PolicyEvaluator,
    lru_faults: int,
    belady_faults: int,
) -> dict:
    """Run CACHEUS once and return metrics."""
    cacheus.reset()
    result = evaluator.run_policy(cacheus, scenario_requests, scenario_name, seed)
    norm = compute_normalized_fault_rate(
        result.metrics.total_faults, belady_faults, lru_faults
    )
    return {
        "scenario": scenario_name,
        "seed": seed,
        "faults": result.metrics.total_faults,
        "norm_rate": norm,
    }


def sweep_hyperparameters(
    expert_pool: list,
    scenarios: dict,
    seeds: list[int],
    evaluator: PolicyEvaluator,
    learning_rates: list[float],
    window_sizes: list[int],
    baselines: dict,
) -> pd.DataFrame:
    """Grid search over learning rate × window size."""
    results = []

    for lr, window in itertools.product(learning_rates, window_sizes):
        scenario_norms = []
        for scenario_name, requests in scenarios.items():
            for seed in seeds:
                # Fresh CACHEUS per (lr, window, scenario, seed)
                cacheus = CACHEUSSelector(
                    experts=expert_pool,
                    learning_rate=lr,
                    window_size=window,
                )
                base = baselines[(scenario_name, seed)]
                metrics = evaluate_cacheus(
                    cacheus, requests, scenario_name, seed,
                    evaluator, base["lru"], base["belady"],
                )
                scenario_norms.append(metrics["norm_rate"])

        results.append({
            "learning_rate": lr,
            "window_size": window,
            "mean_norm_rate": float(np.mean(scenario_norms)),
            "std_norm_rate": float(np.std(scenario_norms)),
            "max_norm_rate": float(np.max(scenario_norms)),
        })
        print(f"  lr={lr}, window={window}: "
              f"mean_norm={np.mean(scenario_norms):.3f}, "
              f"max={np.max(scenario_norms):.3f}")

    return pd.DataFrame(results).sort_values("mean_norm_rate")


def evaluate_pool_compositions(
    ml_experts: list,
    scenarios: dict,
    seeds: list[int],
    evaluator: PolicyEvaluator,
    baselines: dict,
    lr: float,
    window: int,
) -> pd.DataFrame:
    """Compare CACHEUS variants with different expert pool compositions."""
    rows = []
    for pool_name in ["classical_only", "ml_only", "ml_plus_lru", "all_5"]:
        try:
            pool = build_expert_pool(pool_name, ml_experts)
        except Exception as e:
            print(f"  Skipping {pool_name}: {e}")
            continue

        scenario_norms = []
        per_scenario = {}
        for scenario_name, requests in scenarios.items():
            seed_norms = []
            for seed in seeds:
                cacheus = CACHEUSSelector(
                    experts=pool, learning_rate=lr, window_size=window,
                )
                base = baselines[(scenario_name, seed)]
                metrics = evaluate_cacheus(
                    cacheus, requests, scenario_name, seed,
                    evaluator, base["lru"], base["belady"],
                )
                seed_norms.append(metrics["norm_rate"])
            per_scenario[scenario_name] = float(np.mean(seed_norms))
            scenario_norms.extend(seed_norms)

        row = {
            "pool": pool_name,
            "num_experts": len(pool),
            "mean_norm_rate": float(np.mean(scenario_norms)),
            **per_scenario,
        }
        rows.append(row)
        print(f"  {pool_name} ({len(pool)} experts): "
              f"mean_norm={np.mean(scenario_norms):.3f}")

    return pd.DataFrame(rows).sort_values("mean_norm_rate")


def record_trajectory(
    expert_pool: list,
    scenario_requests,
    scenario_name: str,
    seed: int,
    evaluator: PolicyEvaluator,
    lr: float,
    window: int,
) -> dict:
    """Run CACHEUS with trajectory recording and return weight-over-time data."""
    cacheus = CACHEUSSelector(
        experts=expert_pool,
        learning_rate=lr,
        window_size=window,
        record_trajectory=True,
    )
    cacheus.reset()
    evaluator.run_policy(cacheus, scenario_requests, scenario_name, seed)

    if not cacheus.trajectory:
        return {"scenario": scenario_name, "ticks": [], "weights": []}

    ticks = [t for t, _ in cacheus.trajectory]
    weights = np.stack([w for _, w in cacheus.trajectory])
    return {
        "scenario": scenario_name,
        "ticks": ticks,
        "weights": weights,
        "expert_names": cacheus.expert_names,
        "phase_changes": [t for t, _ in cacheus.phase_change_log],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune CACHEUS hyperparameters")
    parser.add_argument("--model-dir", default="data/models")
    parser.add_argument("--output-dir", default="data/results")
    parser.add_argument("--seeds", type=int, default=3,
                        help="Number of seeds per scenario (CACHEUS sweep is expensive)")
    parser.add_argument("--no-predicted-reuse", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(args.model_dir)
    feature_config = FeatureConfig(use_predicted_reuse=not args.no_predicted_reuse)

    seeds = [42, 123, 456, 789, 1337][:args.seeds]
    evaluator = PolicyEvaluator(weight_blocks=64, workspace_blocks=32)

    print("Loading ML experts...")
    ml_experts = load_ml_experts(model_dir, feature_config)
    if not ml_experts:
        print("Error: no trained ML models found in", model_dir)
        return
    print(f"  Loaded {len(ml_experts)} ML experts: "
          f"{[e.name() for e in ml_experts]}")

    print("\nGenerating workloads...")
    scenarios = {}
    for name in WorkloadGenerator.all_scenario_names():
        # Use first seed's workload for sweep speed; trajectories use first seed
        wg = WorkloadGenerator(seed=seeds[0])
        scenarios[name] = wg.generate_scenario(name)

    print("\nComputing LRU and Belady baselines...")
    baselines: dict[tuple[str, int], dict] = {}
    for scenario_name, requests in scenarios.items():
        for seed in seeds:
            wg = WorkloadGenerator(seed=seed)
            scenario_requests = wg.generate_scenario(scenario_name)
            # LRU baseline
            lru_result = evaluator.run_policy(
                LRUPolicy(), scenario_requests, scenario_name, seed,
            )
            # Belady oracle
            oracle = build_belady_oracle(scenario_requests)
            belady_result = evaluator.run_policy(
                oracle, scenario_requests, scenario_name, seed,
            )
            baselines[(scenario_name, seed)] = {
                "lru": lru_result.metrics.total_faults,
                "belady": belady_result.metrics.total_faults,
            }
        print(f"  {scenario_name}: baselines computed")

    # 1. Hyperparameter sweep on the all_5 pool
    print("\n=== Sweep: learning_rate × window_size (all_5 pool) ===")
    expert_pool_all5 = build_expert_pool("all_5", ml_experts)
    sweep_df = sweep_hyperparameters(
        expert_pool=expert_pool_all5,
        scenarios=scenarios,
        seeds=seeds,
        evaluator=evaluator,
        learning_rates=[0.05, 0.1, 0.2, 0.4],
        window_sizes=[50, 100, 200],
        baselines=baselines,
    )
    sweep_path = output_dir / "cacheus_hyperparam_sweep.csv"
    sweep_df.to_csv(sweep_path, index=False)
    print(f"\nSweep results saved: {sweep_path}")
    best = sweep_df.iloc[0]
    print(f"\nBest config: lr={best.learning_rate}, window={best.window_size} "
          f"-> mean_norm={best.mean_norm_rate:.3f}")

    best_lr = float(best.learning_rate)
    best_window = int(best.window_size)

    # 2. Compare expert pool compositions at the best lr/window
    print(f"\n=== Pool composition (lr={best_lr}, window={best_window}) ===")
    pool_df = evaluate_pool_compositions(
        ml_experts=ml_experts,
        scenarios=scenarios,
        seeds=seeds,
        evaluator=evaluator,
        baselines=baselines,
        lr=best_lr,
        window=best_window,
    )
    pool_path = output_dir / "cacheus_pool_comparison.csv"
    pool_df.to_csv(pool_path, index=False)
    print(f"\nPool comparison saved: {pool_path}")

    # 3. Trajectory recording for the best pool on each scenario
    print(f"\n=== Recording weight trajectories (best config) ===")
    best_pool_name = pool_df.iloc[0]["pool"]
    best_pool = build_expert_pool(best_pool_name, ml_experts)
    print(f"  Using pool: {best_pool_name} ({[e.name() for e in best_pool]})")

    trajectories = {}
    for scenario_name in scenarios:
        wg = WorkloadGenerator(seed=seeds[0])
        scenario_requests = wg.generate_scenario(scenario_name)
        traj = record_trajectory(
            expert_pool=best_pool,
            scenario_requests=scenario_requests,
            scenario_name=scenario_name,
            seed=seeds[0],
            evaluator=evaluator,
            lr=best_lr,
            window=best_window,
        )
        trajectories[scenario_name] = traj
        n_decisions = len(traj["ticks"])
        n_phase = len(traj.get("phase_changes", []))
        print(f"  {scenario_name}: {n_decisions} decisions, "
              f"{n_phase} phase changes detected")

    # Persist trajectories as JSON (small enough). Force Python types so
    # numpy int64 / float32 don't slip through and break json.dump.
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
    traj_path = output_dir / "cacheus_trajectories.json"
    with open(traj_path, "w") as f:
        json.dump(traj_serializable, f)
    print(f"\nTrajectories saved: {traj_path}")

    # Final summary
    print("\n=== Summary ===")
    print(f"Best hyperparameters: lr={best_lr}, window={best_window}")
    print(f"Best pool composition: {best_pool_name}")
    print(f"Best mean normalized fault rate: {pool_df.iloc[0]['mean_norm_rate']:.3f}")
    print()
    print("Per-scenario norm rates (best config):")
    for col in pool_df.columns:
        if col not in ("pool", "num_experts", "mean_norm_rate"):
            val = pool_df.iloc[0].get(col)
            if val is not None:
                print(f"  {col}: {val:.3f}")


if __name__ == "__main__":
    main()
