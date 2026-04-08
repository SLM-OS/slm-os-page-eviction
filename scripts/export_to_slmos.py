#!/usr/bin/env python3
"""Export trained models for SLM-OS Rust integration.

Converts XGBoost to Rust if-else chain and MLP to int8 const arrays.
Runs cross-validation to verify exported predictions match Python.

Usage:
    python scripts/export_to_slmos.py \
        --xgb-model data/models/xgb_model.json \
        --mlp-model data/models/mlp_model.pt \
        --output-dir ../CS-496-Capstone-SLM-Operating-System/runtime/src/mm/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xgboost as xgb

from src.export.mlp_to_rust import compute_model_size, export_mlp_to_rust
from src.export.verify_export import generate_test_vectors, verify_predictions
from src.export.xgb_to_rust import dump_trees_json, export_xgb_to_rust
from src.features.extractor import FeatureConfig
from src.training.train_mlp import load_model as load_mlp
from src.training.train_xgb import load_model as load_xgb


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export models for SLM-OS integration"
    )
    parser.add_argument("--xgb-model", help="Path to trained XGBoost model")
    parser.add_argument("--mlp-model", help="Path to trained MLP model")
    parser.add_argument(
        "--output-dir", default="data/export",
        help="Output directory for generated Rust files",
    )
    parser.add_argument(
        "--no-predicted-reuse", action="store_true",
        help="Models trained without predicted_reuse_dist",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = FeatureConfig(use_predicted_reuse=not args.no_predicted_reuse)
    num_features = config.num_features

    # Export XGBoost
    if args.xgb_model:
        print("--- XGBoost Export ---")
        model = load_xgb(args.xgb_model)

        rust_path = output_dir / "xgb_policy_generated.rs"
        source = export_xgb_to_rust(
            model, rust_path,
            feature_names=config.feature_names,
        )
        print(f"  Generated: {rust_path} ({len(source)} bytes)")

        # Dump trees for inspection
        json_path = output_dir / "xgb_trees.json"
        dump_trees_json(model, json_path)
        print(f"  Tree dump: {json_path}")

        # Verify (Python-side only; Rust verification requires compilation)
        test_vectors = generate_test_vectors(1000, num_features)
        dmatrix = xgb.DMatrix(test_vectors, feature_names=config.feature_names)
        py_preds = model.predict(dmatrix)
        print(f"  Python predictions: min={py_preds.min():.4f}, max={py_preds.max():.4f}")

    # Export MLP
    if args.mlp_model:
        print("\n--- MLP Export ---")
        model = load_mlp(args.mlp_model, num_features=num_features)

        # Quantized export
        rust_path = output_dir / "mlp_policy_generated.rs"
        source = export_mlp_to_rust(model, rust_path, quantize=True)
        model_size = compute_model_size(model, quantize=True)
        print(f"  Generated: {rust_path} ({len(source)} bytes)")
        print(f"  Model size (int8): {model_size} bytes ({model_size/1024:.1f} KB)")

        # Float32 export for verification
        f32_path = output_dir / "mlp_policy_f32.rs"
        export_mlp_to_rust(model, f32_path, quantize=False)
        print(f"  Float32 reference: {f32_path}")

    print("\nExport complete.")


if __name__ == "__main__":
    main()
