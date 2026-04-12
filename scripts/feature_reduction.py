#!/usr/bin/env python3
"""Test XGBoost accuracy with reduced feature set (top-10 only).

Also profiles inference latency for XGBoost and MLP on a large batch.

Usage:
    python scripts/feature_reduction.py --data data/traces/eviction_events.parquet
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xgboost as xgb

from src.features.extractor import FeatureConfig
from src.training.dataset import EvictionDataset, load_dataset, train_val_test_split
from src.training.train_xgb import (
    XGBTrainConfig,
    load_model as load_xgb,
    train_xgboost,
)
from src.training.train_mlp import load_model as load_mlp


# Top 10 features by XGBoost gain (from prior analysis)
TOP_10_FEATURES = [
    "predicted_reuse_dist",
    "time_since_access",
    "time_since_load",
    "model_priority",
    "req_block_model_id",
    "recency_rank",
    "num_loaded_models",
    "access_count",
    "req_block_priority",
    "frequency_rank",
]


def _eviction_accuracy(preds, is_optimal, eviction_ids):
    """Compute eviction decision accuracy (argmax agreement per eviction)."""
    df = pd.DataFrame({
        "pred": preds,
        "is_optimal": is_optimal,
        "eviction_id": eviction_ids,
    })
    grouped = df.groupby("eviction_id")
    pred_choice = grouped["pred"].idxmax()
    true_choice = grouped["is_optimal"].idxmax()
    return float((pred_choice == true_choice).mean())


def select_features(
    data: EvictionDataset,
    feature_names: list[str],
) -> EvictionDataset:
    """Return a new EvictionDataset with only the selected feature columns."""
    col_indices = [data.feature_names.index(name) for name in feature_names]
    return EvictionDataset(
        features=data.features[:, col_indices].copy(),
        is_optimal=data.is_optimal,
        reuse_distance=data.reuse_distance,
        eviction_ids=data.eviction_ids,
        scenarios=data.scenarios,
        feature_names=feature_names,
    )


def test_reduced_features(
    train_data: EvictionDataset,
    val_data: EvictionDataset,
    test_data: EvictionDataset,
) -> None:
    """Train XGBoost on top-10 features and compare accuracy."""
    print("\n=== Reduced Feature Set Experiment ===")
    print(f"Full features: {len(train_data.feature_names)}")
    print(f"Reduced features: {len(TOP_10_FEATURES)}")
    print(f"Selected: {TOP_10_FEATURES}")

    # Select top 10 features
    train_sub = select_features(train_data, TOP_10_FEATURES)
    val_sub = select_features(val_data, TOP_10_FEATURES)
    test_sub = select_features(test_data, TOP_10_FEATURES)

    # Train on reduced set
    config = XGBTrainConfig()
    result = train_xgboost(train_sub, val_sub, config)
    print(f"\nReduced-feature XGBoost:")
    print(f"  Val AUC: {result.val_auc:.6f}")
    print(f"  Best iteration: {result.best_iteration}")

    # Evaluate on test set
    dtest = xgb.DMatrix(test_sub.features, feature_names=test_sub.feature_names)
    preds = result.model.predict(dtest)
    accuracy = _eviction_accuracy(preds, test_sub.is_optimal, test_sub.eviction_ids)

    print(f"  Test accuracy: {accuracy:.4f}")
    print(f"  Full-feature baseline: 0.9607 (from analyze_models.py)")
    print(f"  Delta: {accuracy - 0.9607:+.4f}")


def profile_latency(
    test_data: EvictionDataset,
    xgb_model_path: str,
    mlp_model_path: str,
    num_features: int,
) -> None:
    """Profile inference latency for XGBoost and MLP."""
    print("\n=== Inference Latency Profiling ===")

    # Sample batch sizes matching realistic eviction candidate counts
    batch_sizes = [16, 32, 64]

    # Load models
    booster = load_xgb(xgb_model_path)
    mlp = load_mlp(mlp_model_path, num_features)

    # Use random test samples for timing
    rng = np.random.default_rng(42)

    for batch_size in batch_sizes:
        print(f"\nBatch size: {batch_size} candidates")
        # Sample random rows
        num_trials = 1000
        indices = rng.choice(test_data.num_rows, size=batch_size * num_trials, replace=True)
        samples = test_data.features[indices].reshape(num_trials, batch_size, -1)

        # XGBoost
        xgb_times = []
        for i in range(num_trials):
            batch = samples[i]
            start = time.perf_counter()
            dm = xgb.DMatrix(batch, feature_names=test_data.feature_names)
            _ = booster.predict(dm)
            xgb_times.append(time.perf_counter() - start)

        xgb_p50 = np.percentile(xgb_times, 50) * 1e6  # us
        xgb_p99 = np.percentile(xgb_times, 99) * 1e6
        xgb_mean = np.mean(xgb_times) * 1e6

        # MLP
        mlp_times = []
        with torch.no_grad():
            for i in range(num_trials):
                batch = samples[i]
                start = time.perf_counter()
                tensor = torch.from_numpy(batch).float()
                _ = mlp.predict_scores(tensor)
                mlp_times.append(time.perf_counter() - start)

        mlp_p50 = np.percentile(mlp_times, 50) * 1e6
        mlp_p99 = np.percentile(mlp_times, 99) * 1e6
        mlp_mean = np.mean(mlp_times) * 1e6

        print(f"  XGBoost (Python): p50={xgb_p50:.1f}us, p99={xgb_p99:.1f}us, "
              f"mean={xgb_mean:.1f}us")
        print(f"  MLP (PyTorch):    p50={mlp_p50:.1f}us, p99={mlp_p99:.1f}us, "
              f"mean={mlp_mean:.1f}us")
        print(f"  Per-candidate XGBoost: {xgb_mean/batch_size:.2f}us")
        print(f"  Per-candidate MLP:     {mlp_mean/batch_size:.2f}us")

    print("\nNote: Python overhead dominates these timings.")
    print("Rust-exported XGBoost (if-else chains) expected to be sub-microsecond per decision.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Feature reduction + latency profiling")
    parser.add_argument(
        "--data", default="data/traces/eviction_events.parquet",
    )
    parser.add_argument(
        "--model-dir", default="data/models",
    )
    parser.add_argument(
        "--no-predicted-reuse", action="store_true",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    config = FeatureConfig(use_predicted_reuse=not args.no_predicted_reuse)

    print("Loading dataset...")
    dataset = load_dataset(args.data, config)
    train_data, val_data, test_data = train_val_test_split(dataset)
    print(f"  Train: {train_data.num_rows}, Val: {val_data.num_rows}, "
          f"Test: {test_data.num_rows}")

    # Reduced feature experiment
    test_reduced_features(train_data, val_data, test_data)

    # Latency profiling
    suffix = "" if config.use_predicted_reuse else "_no_reuse"
    xgb_path = model_dir / f"xgb_model{suffix}.json"
    mlp_path = model_dir / f"mlp_model{suffix}.pt"

    if xgb_path.exists() and mlp_path.exists():
        profile_latency(test_data, str(xgb_path), str(mlp_path), config.num_features)

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
