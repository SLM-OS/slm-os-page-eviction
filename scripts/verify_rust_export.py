#!/usr/bin/env python3
"""End-to-end verification of the Python → Rust export pipeline.

Generates standalone Rust code, compiles it with rustc, runs it on
random test vectors, and compares its predictions against the Python
reference. This is the Phase 6.1 verification step.

Usage:
    python scripts/verify_rust_export.py --model-dir data/models/ \\
                                         --output-dir data/export/verify/
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import xgboost as xgb

from src.export.mlp_to_rust import export_mlp_to_rust
from src.export.verify_export import verify_predictions
from src.export.xgb_to_rust import export_xgb_to_rust
from src.features.extractor import FeatureConfig
from src.training.train_mlp import load_model as load_mlp
from src.training.train_xgb import load_model as load_xgb


XGB_HARNESS = """\
{xgb_module}

fn main() {{
    use std::io::BufRead;
    let stdin = std::io::stdin();
    for line in stdin.lock().lines() {{
        let line = line.expect("read line");
        if line.trim().is_empty() {{ continue; }}
        let nums: Vec<f32> = line.split(',')
            .map(|s| s.trim().parse::<f32>().expect("parse f32"))
            .collect();
        assert_eq!(nums.len(), {num_features},
            "expected {num_features} features, got {{}}", nums.len());
        let mut arr = [0.0_f32; {num_features}];
        arr.copy_from_slice(&nums);
        println!("{{}}", xgb_predict(&arr));
    }}
}}
"""


MLP_HARNESS = """\
{mlp_module}

fn main() {{
    use std::io::BufRead;
    let stdin = std::io::stdin();
    for line in stdin.lock().lines() {{
        let line = line.expect("read line");
        if line.trim().is_empty() {{ continue; }}
        let nums: Vec<f32> = line.split(',')
            .map(|s| s.trim().parse::<f32>().expect("parse f32"))
            .collect();
        assert_eq!(nums.len(), {num_features},
            "expected {num_features} features, got {{}}", nums.len());
        let mut arr = [0.0_f32; {num_features}];
        arr.copy_from_slice(&nums);
        println!("{{}}", mlp_predict(&arr));
    }}
}}
"""


def _check_rustc() -> bool:
    if shutil.which("rustc") is None:
        print("Error: rustc not in PATH. Install Rust to run verification.")
        return False
    return True


def _compile_and_run(
    rust_source: str,
    workdir: Path,
    binary_name: str,
    stdin_input: str,
) -> str:
    """Write source, compile with rustc, and run with stdin → captured stdout."""
    workdir.mkdir(parents=True, exist_ok=True)
    src_path = workdir / f"{binary_name}.rs"
    src_path.write_text(rust_source)

    bin_path = workdir / binary_name
    print(f"  Compiling {src_path.name} ... ", end="", flush=True)
    res = subprocess.run(
        ["rustc", "-O", str(src_path), "-o", str(bin_path)],
        capture_output=True, text=True,
    )
    if res.returncode != 0:
        print("FAILED")
        print(res.stderr)
        sys.exit(1)
    print(f"OK ({bin_path.stat().st_size // 1024} KB binary)")

    print(f"  Running {bin_path.name} ... ", end="", flush=True)
    res = subprocess.run(
        [str(bin_path)], input=stdin_input,
        capture_output=True, text=True,
    )
    if res.returncode != 0:
        print("FAILED")
        print(res.stderr)
        sys.exit(1)
    print("OK")
    return res.stdout.strip()


def _to_csv_lines(arr: np.ndarray) -> str:
    """Serialize a 2D numpy array as CSV: one row per line."""
    return "\n".join(",".join(repr(float(v)) for v in row) for row in arr) + "\n"


def _parse_lines(stdout: str) -> np.ndarray:
    """Parse Rust output: one float per line."""
    return np.array(
        [float(line) for line in stdout.splitlines() if line.strip()],
        dtype=np.float32,
    )


def verify_xgboost(
    booster: xgb.Booster,
    feature_names: list[str],
    output_dir: Path,
    test_vectors: np.ndarray,
) -> None:
    """Verify XGBoost Python ↔ Rust agreement."""
    print("\n--- XGBoost ---")

    # 1. Generate standalone Rust module
    rust_module = export_xgb_to_rust(
        booster,
        output_dir / "xgb_module.rs",
        feature_names=feature_names,
        standalone=True,
    )

    # 2. Wrap in main()
    harness = XGB_HARNESS.format(
        xgb_module=rust_module,
        num_features=len(feature_names),
    )

    # 3. Compile + run
    stdin = _to_csv_lines(test_vectors)
    rust_output = _compile_and_run(
        harness, output_dir, "xgb_verify", stdin,
    )
    rust_preds = _parse_lines(rust_output)

    # 4. Python reference
    dmatrix_py = xgb.DMatrix(test_vectors, feature_names=feature_names)
    py_preds = booster.predict(dmatrix_py)

    # 5. Compare
    result = verify_predictions(py_preds, rust_preds, tolerance=1e-3)
    print(f"  Test vectors: {len(test_vectors)}")
    print(f"  Max abs error: {result.max_abs_error:.6f}")
    print(f"  Mean abs error: {result.mean_abs_error:.6f}")
    print(f"  Mismatches at tolerance=1e-3: {result.num_mismatches}")
    print(f"  {'PASS' if result.passed else 'FAIL'}")


def verify_mlp(
    mlp,
    num_features: int,
    output_dir: Path,
    test_vectors: np.ndarray,
) -> None:
    """Verify MLP Python ↔ Rust (int8 quantized) agreement."""
    print("\n--- MLP (int8 quantized) ---")

    # 1. Generate standalone Rust module
    rust_module = export_mlp_to_rust(
        mlp,
        output_dir / "mlp_module.rs",
        quantize=True,
    )

    # 2. Wrap in main()
    harness = MLP_HARNESS.format(
        mlp_module=rust_module,
        num_features=num_features,
    )

    # 3. Compile + run
    stdin = _to_csv_lines(test_vectors)
    rust_output = _compile_and_run(
        harness, output_dir, "mlp_verify", stdin,
    )
    rust_preds = _parse_lines(rust_output)

    # 4. Python reference (float32 — int8 will differ within quantization tolerance)
    import torch
    with torch.no_grad():
        tensor = torch.from_numpy(test_vectors).float()
        py_preds = mlp.predict_scores(tensor).numpy()

    # 5. Per-prediction comparison. Int8 quantization introduces real per-sample
    #    error (up to ~0.2 on extreme cases) — what matters in practice is
    #    decision agreement (argmax within an eviction group). Here we report
    #    raw error stats and a synthetic "argmax stability" check on random
    #    candidate groups.
    abs_err = np.abs(py_preds - rust_preds)
    print(f"  Test vectors: {len(test_vectors)}")
    print(f"  Max abs error: {abs_err.max():.6f}")
    print(f"  Mean abs error: {abs_err.mean():.6f}")
    print(f"  Median abs error: {np.median(abs_err):.6f}")
    print(f"  95th-percentile error: {np.percentile(abs_err, 95):.6f}")

    # Decision-agreement check: chunk vectors into synthetic 8-candidate groups
    # and check argmax agreement (matches how the policy is used at runtime).
    group_size = 8
    n_groups = len(test_vectors) // group_size
    if n_groups > 0:
        py_groups = py_preds[: n_groups * group_size].reshape(n_groups, group_size)
        rust_groups = rust_preds[: n_groups * group_size].reshape(n_groups, group_size)
        agree = (py_groups.argmax(axis=1) == rust_groups.argmax(axis=1)).sum()
        decision_acc = agree / n_groups
        print(f"  Decision agreement on {n_groups} groups of {group_size}: "
              f"{agree}/{n_groups} = {decision_acc:.4f}")
        passed = decision_acc >= 0.95
        print(f"  {'PASS' if passed else 'FAIL'} (target >= 0.95)")
    else:
        passed = abs_err.mean() < 0.05
        print(f"  {'PASS' if passed else 'FAIL'} (mean error < 0.05)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify Python ↔ Rust export agreement")
    parser.add_argument("--model-dir", default="data/models")
    parser.add_argument("--output-dir", default="data/export/verify")
    parser.add_argument("--num-vectors", type=int, default=500,
                        help="Number of random test vectors")
    parser.add_argument("--no-predicted-reuse", action="store_true")
    args = parser.parse_args()

    if not _check_rustc():
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(args.model_dir)
    config = FeatureConfig(use_predicted_reuse=not args.no_predicted_reuse)

    # Reproducible random vectors in [0, 1] (matching normalized feature space)
    rng = np.random.default_rng(42)
    test_vectors = rng.uniform(0, 1, size=(args.num_vectors, config.num_features)) \
                      .astype(np.float32)

    suffix = "" if config.use_predicted_reuse else "_no_reuse"
    xgb_path = model_dir / f"xgb_model{suffix}.json"
    mlp_path = model_dir / f"mlp_model{suffix}.pt"

    if xgb_path.exists():
        booster = load_xgb(str(xgb_path))
        verify_xgboost(booster, config.feature_names, output_dir, test_vectors)
    else:
        print(f"Skipping XGBoost: {xgb_path} not found")

    if mlp_path.exists():
        mlp = load_mlp(str(mlp_path), config.num_features)
        verify_mlp(mlp, config.num_features, output_dir, test_vectors)
    else:
        print(f"Skipping MLP: {mlp_path} not found")

    print("\nDone.")


if __name__ == "__main__":
    main()
