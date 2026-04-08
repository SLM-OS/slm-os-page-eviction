#!/usr/bin/env python3
"""Train all models: XGBoost and MLP.

Loads the generated dataset, trains both models with the specified
configurations, and saves checkpoints to data/models/.

Usage:
    python scripts/train_all.py [--dataset data/traces/eviction_events.parquet]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.features.extractor import FeatureConfig
from src.training.dataset import load_dataset, train_val_test_split
from src.training.train_xgb import (
    XGBTrainConfig,
    cross_validate_xgb,
    save_model as save_xgb,
    train_xgboost,
)
from src.training.train_mlp import (
    MLPTrainConfig,
    save_model as save_mlp,
    train_mlp,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train all models")
    parser.add_argument(
        "--dataset", default="data/traces/eviction_events.parquet",
        help="Path to training dataset",
    )
    parser.add_argument(
        "--output-dir", default="data/models",
        help="Output directory for model checkpoints",
    )
    parser.add_argument(
        "--no-predicted-reuse", action="store_true",
        help="Train without predicted_reuse_dist feature",
    )
    parser.add_argument(
        "--skip-xgb", action="store_true", help="Skip XGBoost training",
    )
    parser.add_argument(
        "--skip-mlp", action="store_true", help="Skip MLP training",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = FeatureConfig(use_predicted_reuse=not args.no_predicted_reuse)

    # Load and split dataset
    print("Loading dataset...")
    dataset = load_dataset(args.dataset, config)
    print(
        f"  Rows: {dataset.num_rows}, "
        f"Evictions: {dataset.num_evictions}, "
        f"Features: {dataset.num_features}, "
        f"Class balance: {dataset.class_balance:.4f}"
    )

    train_data, val_data, test_data = train_val_test_split(dataset)
    print(
        f"  Train: {train_data.num_rows}, "
        f"Val: {val_data.num_rows}, "
        f"Test: {test_data.num_rows}"
    )

    # Train XGBoost
    if not args.skip_xgb:
        print("\n--- XGBoost Training ---")
        xgb_config = XGBTrainConfig()
        result = train_xgboost(train_data, val_data, xgb_config)
        print(f"  Train AUC: {result.train_auc:.4f}")
        print(f"  Val AUC: {result.val_auc:.4f}")
        print(f"  Best iteration: {result.best_iteration}")

        suffix = "" if config.use_predicted_reuse else "_no_reuse"
        save_xgb(result.model, output_dir / f"xgb_model{suffix}.json")
        print(f"  Saved: {output_dir / f'xgb_model{suffix}.json'}")

        # Cross-validation
        print("\n  Cross-validation...")
        cv_aucs = cross_validate_xgb(dataset, xgb_config, n_folds=5)
        print(f"  CV AUCs: {[f'{a:.4f}' for a in cv_aucs]}")
        print(f"  Mean: {sum(cv_aucs)/len(cv_aucs):.4f}")

        # Feature importance
        print("\n  Top features by gain:")
        sorted_imp = sorted(
            result.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        for name, gain in sorted_imp[:10]:
            print(f"    {name}: {gain:.2f}")

    # Train MLP
    if not args.skip_mlp:
        print("\n--- MLP Training ---")
        mlp_config = MLPTrainConfig(num_features=config.num_features)
        result = train_mlp(train_data, val_data, mlp_config)
        print(f"  Best epoch: {result.best_epoch}")
        print(f"  Best val loss: {result.best_val_loss:.4f}")

        suffix = "" if config.use_predicted_reuse else "_no_reuse"
        save_mlp(result.model, output_dir / f"mlp_model{suffix}.pt")
        print(f"  Saved: {output_dir / f'mlp_model{suffix}.pt'}")

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
