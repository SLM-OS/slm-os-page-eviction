#!/usr/bin/env python3
"""Comprehensive model analysis for Phases 3-4.

Covers:
- Feature importance analysis with gain ranking
- MLP quantization accuracy verification
- XGBoost and MLP test set evaluation (eviction decision accuracy)
- DAgger improvement measurement

Usage:
    python scripts/analyze_models.py --data data/traces/eviction_events.parquet \
                                     --model-dir data/models/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from src.features.extractor import FeatureConfig
from src.training.dataset import EvictionDataset, load_dataset, train_val_test_split
from src.training.train_xgb import load_model as load_xgb
from src.training.train_mlp import (
    PageReplacementMLP,
    load_model as load_mlp,
)
from src.export.mlp_to_rust import quantize_model


def analyze_feature_importance(
    model_path: str,
    feature_names: list[str],
) -> None:
    """Analyze XGBoost feature importance (gain-based)."""
    import xgboost as xgb

    print("\n=== Feature Importance Analysis ===")
    model = load_xgb(model_path)
    importance = model.get_score(importance_type="gain")

    # Sort by gain
    sorted_feats = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    total_gain = sum(v for _, v in sorted_feats)

    print(f"\nAll {len(sorted_feats)} features ranked by gain:")
    print(f"{'Rank':<6}{'Feature':<30}{'Gain':<12}{'% of Total':<12}")
    print("-" * 60)
    for rank, (feat, gain) in enumerate(sorted_feats, 1):
        pct = 100.0 * gain / total_gain
        print(f"{rank:<6}{feat:<30}{gain:<12.1f}{pct:<12.1f}%")

    # Top 10 cumulative
    top10_gain = sum(v for _, v in sorted_feats[:10])
    print(f"\nTop 10 features account for {100*top10_gain/total_gain:.1f}% of total gain")

    # Features not used
    unused = set(feature_names) - set(importance.keys())
    if unused:
        print(f"\nUnused features ({len(unused)}): {sorted(unused)}")


def _eviction_decision_accuracy(
    preds: np.ndarray,
    is_optimal: np.ndarray,
    eviction_ids: np.ndarray,
    scenarios: np.ndarray,
) -> tuple[float, dict[str, float]]:
    """Vectorized eviction decision accuracy using pandas groupby."""
    import pandas as pd

    df = pd.DataFrame({
        "pred": preds,
        "is_optimal": is_optimal,
        "eviction_id": eviction_ids,
        "scenario": scenarios,
    })

    # For each eviction, find argmax of pred and argmax of is_optimal
    # Use idxmax within each group
    grouped = df.groupby("eviction_id")
    pred_choice = grouped["pred"].idxmax()
    true_choice = grouped["is_optimal"].idxmax()
    correct = (pred_choice == true_choice)

    # Get scenario for each eviction_id
    scenario_per_id = df.groupby("eviction_id")["scenario"].first()

    overall = correct.mean()
    per_scenario = {}
    for scenario in sorted(scenario_per_id.unique()):
        mask = scenario_per_id == scenario
        per_scenario[scenario] = float(correct[mask].mean())

    return float(overall), per_scenario


def evaluate_eviction_accuracy(
    test_data: EvictionDataset,
    xgb_model_path: str | None,
    mlp_model_path: str | None,
    num_features: int,
) -> dict[str, dict[str, float]]:
    """Evaluate eviction decision accuracy on test set."""
    import xgboost as xgb

    print("\n=== Eviction Decision Accuracy (Test Set) ===")
    results = {}

    unique_ids = np.unique(test_data.eviction_ids)
    print(f"Test set: {test_data.num_rows} candidates, {len(unique_ids)} eviction events")
    print(f"Test scenarios: {sorted(set(test_data.scenarios))}")

    for model_name, model_path in [
        ("XGBoost", xgb_model_path),
        ("MLP", mlp_model_path),
    ]:
        if model_path is None or not Path(model_path).exists():
            continue

        if model_name == "XGBoost":
            booster = load_xgb(model_path)
            dtest = xgb.DMatrix(
                test_data.features,
                feature_names=test_data.feature_names,
            )
            preds = booster.predict(dtest)
        else:
            model = load_mlp(model_path, num_features)
            with torch.no_grad():
                tensor = torch.from_numpy(test_data.features).float()
                preds = model.predict_scores(tensor).numpy()

        accuracy, per_scenario = _eviction_decision_accuracy(
            preds, test_data.is_optimal, test_data.eviction_ids, test_data.scenarios,
        )

        print(f"\n{model_name}:")
        print(f"  Overall: {accuracy:.4f}")
        for scenario, acc in per_scenario.items():
            print(f"  {scenario}: {acc:.4f}")

        results[model_name] = {"accuracy": accuracy}

    return results


def verify_quantization(
    model_path: str,
    test_data: EvictionDataset,
    num_features: int,
) -> None:
    """Verify int8 quantization accuracy loss."""
    print("\n=== MLP Quantization Verification ===")

    model = load_mlp(model_path, num_features)
    quantized = quantize_model(model)

    # Run float32 model
    with torch.no_grad():
        tensor = torch.from_numpy(test_data.features).float()
        float_preds = model.predict_scores(tensor).numpy()

    # Simulate quantized inference in Python
    state = model.state_dict()
    layers = [
        ("shared.0", "shared.0"),
        ("shared.3", "shared.3"),
        ("shared.6", "shared.6"),
        ("classify_head.0", "classify_head.0"),
    ]

    # Forward pass with quantized weights
    x = test_data.features.copy()
    for i, (w_key_base, b_key_base) in enumerate(layers):
        w_key = f"{w_key_base}.weight"
        b_key = f"{w_key_base}.bias"

        w_q = quantized[f"{w_key}_q"]
        w_scale = quantized[f"{w_key}_scale"][0]
        b_q = quantized[f"{b_key}_q"]
        b_scale = quantized[f"{b_key}_scale"][0]

        # Dequantize
        w_deq = w_q.astype(np.float32) * w_scale
        b_deq = b_q.astype(np.float32) * b_scale

        x = x @ w_deq.T + b_deq
        if i < len(layers) - 1:
            x = np.maximum(x, 0)  # ReLU

    # Sigmoid
    quant_preds = 1.0 / (1.0 + np.exp(-x.squeeze()))

    # Compare
    abs_errors = np.abs(float_preds - quant_preds)
    print(f"  Max absolute error: {abs_errors.max():.6f}")
    print(f"  Mean absolute error: {abs_errors.mean():.6f}")
    print(f"  Median absolute error: {np.median(abs_errors):.6f}")
    print(f"  95th percentile error: {np.percentile(abs_errors, 95):.6f}")

    # Check eviction decision agreement (vectorized)
    import pandas as pd
    df = pd.DataFrame({
        "float_pred": float_preds,
        "quant_pred": quant_preds,
        "eviction_id": test_data.eviction_ids,
    })
    grouped = df.groupby("eviction_id")
    float_choice = grouped["float_pred"].idxmax()
    quant_choice = grouped["quant_pred"].idxmax()
    agree = int((float_choice == quant_choice).sum())
    num_events = len(grouped)

    decision_accuracy = agree / num_events if num_events > 0 else 0
    print(f"\n  Eviction decision agreement: {agree}/{num_events} = {decision_accuracy:.4f}")
    print(f"  Target: >= 0.99")
    print(f"  {'PASS' if decision_accuracy >= 0.99 else 'FAIL'}: "
          f"accuracy loss = {1.0 - decision_accuracy:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze trained models")
    parser.add_argument(
        "--data", default="data/traces/eviction_events.parquet",
        help="Path to dataset",
    )
    parser.add_argument(
        "--model-dir", default="data/models",
        help="Directory with trained models",
    )
    parser.add_argument(
        "--no-predicted-reuse", action="store_true",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    config = FeatureConfig(use_predicted_reuse=not args.no_predicted_reuse)

    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset(args.data, config)
    train_data, val_data, test_data = train_val_test_split(dataset)
    print(f"  Train: {train_data.num_rows}, Val: {val_data.num_rows}, "
          f"Test: {test_data.num_rows}")

    suffix = "" if config.use_predicted_reuse else "_no_reuse"
    xgb_path = model_dir / f"xgb_model{suffix}.json"
    mlp_path = model_dir / f"mlp_model{suffix}.pt"

    # Feature importance
    if xgb_path.exists():
        analyze_feature_importance(str(xgb_path), config.feature_names)

    # Eviction decision accuracy
    evaluate_eviction_accuracy(
        test_data,
        str(xgb_path) if xgb_path.exists() else None,
        str(mlp_path) if mlp_path.exists() else None,
        config.num_features,
    )

    # Quantization verification
    if mlp_path.exists():
        verify_quantization(str(mlp_path), test_data, config.num_features)

    print("\n=== Analysis Complete ===")


if __name__ == "__main__":
    main()
