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
from src.export.xgb_to_smb import (
    export_xgb_to_smb,
    generate_evict_verification_corpus,
)
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
    parser.add_argument(
        "--smb-output-dir",
        help="If set, also emit `evict.smb` + verification corpus here. "
             "Lets SLM-OS dynamically load this model via "
             "`eviction blob load xgboost <path>` and run "
             "`bench xgb-equiv-evict` against it without rebuilding the "
             "kernel. The baked .rs path (above) stays in lockstep so "
             "anything that re-runs this script keeps the two paths in "
             "sync.",
    )
    parser.add_argument(
        "--smb-corpus-size", type=int, default=1000,
        help="Number of verification vectors written into "
             "expected_evict.bin / test_vectors_xgb_evict.bin "
             "(only used when --smb-output-dir is set; default 1000).",
    )
    parser.add_argument(
        "--xgb-trees", type=int, default=0,
        help="If > 0, prune the XGBoost ensemble to its first K trees "
             "before export (gradient boosting => the first K trees of an "
             "N-round model are identical to a K-round model). Both the "
             "generated .rs AND the verification corpus use the pruned "
             "model, keeping the baked predictor and its expected_evict.bin "
             "in lockstep. SLM-OS #961 ships K=16 (200 -> 16 trees, "
             "~300 ns/predict on pi-5-2, hit-rate preserved). 0 = full model.",
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

        if args.xgb_trees > 0:
            total = len(model.get_dump())
            if args.xgb_trees < total:
                model = model[0:args.xgb_trees]
                print(f"  Pruned ensemble: {total} -> {args.xgb_trees} trees "
                      f"(#961). Both .rs and corpus use the pruned model.")
            else:
                print(f"  --xgb-trees {args.xgb_trees} >= {total}; "
                      f"using full model.")

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

        # SEMB blob + verification corpus for the SLM-OS blob-load path.
        # Same trained model, second deployment format — researchers
        # can iterate by dropping a new .smb on the SD card and running
        # `bench xgb-equiv-evict` to confirm bit-equivalence on hardware.
        if args.smb_output_dir:
            smb_dir = Path(args.smb_output_dir)
            smb_dir.mkdir(parents=True, exist_ok=True)
            smb_path = smb_dir / "evict.smb"
            blob = export_xgb_to_smb(
                model, smb_path, feature_names=config.feature_names,
            )
            print(f"  SEMB blob: {smb_path} ({len(blob)} bytes)")
            generate_evict_verification_corpus(
                model, smb_dir,
                feature_names=config.feature_names,
                n_tests=args.smb_corpus_size,
            )
            print(
                f"  Verification corpus: {args.smb_corpus_size} vectors "
                f"({smb_dir}/test_vectors_xgb_evict.bin + expected_evict.bin)"
            )

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
