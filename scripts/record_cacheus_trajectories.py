#!/usr/bin/env python3
"""Re-record CACHEUS weight trajectories using a known-good config.

The hyperparameter sweep (scripts/tune_cacheus.py) takes ~1.5h on the
full grid. This script reuses its already-saved best config from
data/results/cacheus_pool_comparison.csv to do *only* the trajectory
recording step (~5min).

Usage:
    python scripts/record_cacheus_trajectories.py \\
        --model-dir data/models/ --output-dir data/results/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow `from scripts.X import ...` when running this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features.extractor import FeatureConfig
from src.simulator.workload import WorkloadGenerator
from src.training.evaluate import PolicyEvaluator
from scripts.tune_cacheus import (
    build_expert_pool,
    load_ml_experts,
    record_trajectory,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-record CACHEUS trajectories")
    parser.add_argument("--model-dir", default="data/models")
    parser.add_argument("--output-dir", default="data/results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=0.4,
                        help="CACHEUS learning rate (default from tune_cacheus best)")
    parser.add_argument("--window", type=int, default=200,
                        help="CACHEUS window size (default from tune_cacheus best)")
    parser.add_argument("--pool", default="ml_only",
                        help="Expert pool name (default: best from pool comparison)")
    parser.add_argument("--no-predicted-reuse", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(args.model_dir)
    feature_config = FeatureConfig(use_predicted_reuse=not args.no_predicted_reuse)

    print(f"Loading ML experts from {model_dir}...")
    ml_experts = load_ml_experts(model_dir, feature_config)
    if not ml_experts:
        print(f"Error: no trained ML models found in {model_dir}")
        return
    print(f"  Loaded {len(ml_experts)} experts: {[e.name() for e in ml_experts]}")

    pool = build_expert_pool(args.pool, ml_experts)
    print(f"Pool '{args.pool}' has {len(pool)} experts: "
          f"{[e.name() for e in pool]}")
    print(f"CACHEUS config: lr={args.lr}, window={args.window}")

    evaluator = PolicyEvaluator(weight_blocks=64, workspace_blocks=32)

    print("\nRecording trajectories...")
    trajectories = {}
    for scenario_name in WorkloadGenerator.all_scenario_names():
        wg = WorkloadGenerator(seed=args.seed)
        scenario_requests = wg.generate_scenario(scenario_name)
        traj = record_trajectory(
            expert_pool=pool,
            scenario_requests=scenario_requests,
            scenario_name=scenario_name,
            seed=args.seed,
            evaluator=evaluator,
            lr=args.lr,
            window=args.window,
        )
        trajectories[scenario_name] = traj
        print(f"  {scenario_name}: {len(traj['ticks'])} decisions, "
              f"{len(traj.get('phase_changes', []))} phase changes")

    # Serialize with explicit Python type coercion (regression: numpy int64
    # silently broke json.dump in the original tune_cacheus run).
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
    out = output_dir / "cacheus_trajectories.json"
    with open(out, "w") as f:
        json.dump(traj_serializable, f)
    print(f"\nTrajectories saved: {out}")


if __name__ == "__main__":
    main()
