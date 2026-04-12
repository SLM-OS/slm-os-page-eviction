#!/usr/bin/env python3
"""Run DAgger fine-tuning on the MLP and measure improvement.

Compares baseline MLP vs DAgger-tuned MLP on validation loss and
eviction decision accuracy.

Usage:
    python scripts/run_dagger.py --data data/traces/eviction_events.parquet \
                                 --model-dir data/models/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.features.extractor import FeatureConfig
from src.policies.mlp_policy import MLPPolicy
from src.policies.belady import BeladyOracle
from src.policies.lru import LRUPolicy
from src.simulator.core import MemoryState, SimController
from src.simulator.trace import TraceCollector
from src.simulator.workload import WorkloadGenerator
from src.training.dagger import dagger_finetune, DAggerConfig
from src.training.dataset import load_dataset, train_val_test_split
from src.training.train_mlp import (
    MLPTrainConfig,
    load_model as load_mlp,
    save_model as save_mlp,
)


def evaluate_decision_accuracy(
    model,
    test_data,
) -> float:
    """Compute eviction decision accuracy on test set."""
    with torch.no_grad():
        tensor = torch.from_numpy(test_data.features).float()
        preds = model.predict_scores(tensor).numpy()

    df = pd.DataFrame({
        "pred": preds,
        "is_optimal": test_data.is_optimal,
        "eviction_id": test_data.eviction_ids,
    })
    grouped = df.groupby("eviction_id")
    pred_choice = grouped["pred"].idxmax()
    true_choice = grouped["is_optimal"].idxmax()
    return float((pred_choice == true_choice).mean())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DAgger fine-tuning")
    parser.add_argument(
        "--data", default="data/traces/eviction_events.parquet",
    )
    parser.add_argument(
        "--model-dir", default="data/models",
    )
    parser.add_argument(
        "--no-predicted-reuse", action="store_true",
    )
    parser.add_argument(
        "--dagger-rounds", type=int, default=3,
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    config = FeatureConfig(use_predicted_reuse=not args.no_predicted_reuse)

    suffix = "" if config.use_predicted_reuse else "_no_reuse"
    mlp_path = model_dir / f"mlp_model{suffix}.pt"

    if not mlp_path.exists():
        print(f"Error: MLP model not found at {mlp_path}")
        return

    # Load dataset and model
    print("Loading dataset...")
    dataset = load_dataset(args.data, config)
    train_data, val_data, test_data = train_val_test_split(dataset)
    print(f"  Train: {train_data.num_rows}, Val: {val_data.num_rows}, "
          f"Test: {test_data.num_rows}")

    baseline_model = load_mlp(str(mlp_path), config.num_features)

    # Baseline accuracy
    baseline_acc = evaluate_decision_accuracy(baseline_model, test_data)
    print(f"\nBaseline MLP test accuracy: {baseline_acc:.4f}")

    # Prepare DAgger: need scenario requests + Belady oracle
    # Use training scenarios (not test scenarios) for on-policy collection
    print("\nPreparing DAgger scenarios...")
    train_scenarios = ["single_inference", "multi_model", "burst_load",
                       "mixed_priority", "adversarial"]

    # Collect requests from first training scenario with seed 42
    wg = WorkloadGenerator(seed=42)
    all_requests = []
    for scenario in train_scenarios[:3]:  # Use 3 scenarios to limit compute
        requests = wg.generate_scenario(scenario)
        all_requests.extend(requests)
    # Sort by tick
    all_requests.sort(key=lambda r: r.tick)

    # Build Belady oracle from LRU trace
    memory = MemoryState(64, 32)
    trace = TraceCollector()
    lru_sim = SimController(memory=memory, policy=LRUPolicy(), trace_collector=trace)
    lru_sim.run(all_requests)
    oracle = BeladyOracle(future_accesses=trace.access_events)

    # Create MLP policy
    mlp_policy = MLPPolicy(load_mlp(str(mlp_path), config.num_features), config)

    # Run DAgger
    print(f"\nRunning DAgger ({args.dagger_rounds} rounds)...")
    dagger_config = DAggerConfig(
        num_rounds=args.dagger_rounds,
        mlp_config=MLPTrainConfig(num_features=config.num_features),
        mix_ratio=0.5,
    )

    dagger_memory = MemoryState(64, 32)
    result = dagger_finetune(
        mlp_policy=mlp_policy,
        oracle=oracle,
        original_train=train_data,
        val_data=val_data,
        scenario_requests=all_requests,
        memory=dagger_memory,
        config=dagger_config,
    )

    # Report results
    print(f"\nDAgger Results:")
    print(f"  Final dataset size: {result.final_dataset_size}")
    for i, val_loss in enumerate(result.improvement_per_round):
        print(f"  Round {i+1} val loss: {val_loss:.6f}")

    # Evaluate DAgger-tuned model
    dagger_acc = evaluate_decision_accuracy(
        result.round_results[-1].model, test_data
    )
    print(f"\nBaseline MLP test accuracy: {baseline_acc:.4f}")
    print(f"DAgger MLP test accuracy:   {dagger_acc:.4f}")
    print(f"Improvement:                {dagger_acc - baseline_acc:+.4f}")

    # Save DAgger model
    dagger_path = model_dir / f"mlp_model_dagger{suffix}.pt"
    save_mlp(result.round_results[-1].model, str(dagger_path))
    print(f"\nSaved DAgger model: {dagger_path}")


if __name__ == "__main__":
    main()
