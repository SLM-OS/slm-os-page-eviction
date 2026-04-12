# Getting Started

## Prerequisites

- Python 3.10+ (developed with 3.12)
- Recommended: 4GB RAM minimum for full dataset generation

## Setup

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e .
```

Core dependencies (simulator + features):
- numpy, pandas, pyarrow, scikit-learn, pyyaml, pytest

ML dependencies (training + export):
- xgboost, torch

The simulator and classical policies work without xgboost/torch. ML imports are lazy-loaded.

## Running Tests

```bash
python -m pytest tests/ -v
```

All 86 tests should pass. Core tests (simulator, policies, features, CACHEUS, eviction feedback) run without xgboost/torch; 9 tests under `tests/test_training.py` skip automatically if ML dependencies are missing.

## Quick Start: Generate a Dataset

```bash
# Small test run (1 seed, small pools, fast)
python scripts/generate_dataset.py --seeds 1 --weight-blocks 16 --workspace-blocks 8

# Full dataset (5 seeds, default pools, ~200K rows)
python scripts/generate_dataset.py
```

Output: `data/traces/eviction_events.parquet`

## Quick Start: Train Models

```bash
# Requires: xgboost, torch
python scripts/train_all.py --data data/traces/eviction_events.parquet --output data/models/
```

## Quick Start: Benchmark

```bash
# Classical policies only (no trained models needed)
python scripts/benchmark.py --seeds 3 --output-dir data/results/

# With trained models (auto-detected)
python scripts/benchmark.py --model-dir data/models/ --output-dir data/results/
```

## Quick Start: Export to Rust

```bash
# Requires trained models in data/models/
python scripts/export_to_slmos.py --model-dir data/models/ --output-dir data/export/
```

## Quick Start: Analyze & Profile

```bash
# Feature importance, test-set accuracy, int8 quantization verification
python scripts/analyze_models.py --data data/traces/eviction_events.parquet --model-dir data/models/

# Reduced feature set (top-10) + inference latency profiling
python scripts/feature_reduction.py --data data/traces/eviction_events.parquet --model-dir data/models/

# DAgger fine-tuning with baseline comparison
python scripts/run_dagger.py --data data/traces/eviction_events.parquet --model-dir data/models/ --dagger-rounds 3
```

## Project Layout

```
slm-os-page-sim/
├── src/
│   ├── simulator/     # Core simulator (block, pool, controller, workload, trace, metrics)
│   ├── policies/      # Eviction policies (LRU, LFU, ARC, SLM-heuristic, Belady, MLP, XGB, CACHEUS)
│   ├── features/      # Feature extraction (27-dim vector) and normalization
│   ├── training/      # Dataset loading, XGBoost/MLP training, DAgger, evaluation
│   └── export/        # Rust code generation (XGBoost if-else, MLP int8 arrays)
├── tests/             # 61 unit tests across all modules
├── scripts/           # CLI entry points (generate, train, benchmark, export)
├── configs/           # YAML configs for scenarios, XGBoost params, MLP params
├── docs/              # This documentation
└── data/              # Generated data (gitignored): traces/, models/, results/
```

## Configuration Files

| File | Purpose |
|------|---------|
| `configs/scenarios.yaml` | Workload scenario definitions and parameters |
| `configs/xgb_params.yaml` | XGBoost hyperparameters (depth, rounds, regularization) |
| `configs/mlp_params.yaml` | MLP hyperparameters (hidden sizes, learning rate, DAgger rounds) |
| `pyproject.toml` | Build system, dependencies, pytest config |

## Dual Feature Mode

The system supports two feature configurations:

- **27 features** (default): includes `predicted_reuse_dist` heuristic
- **26 features** (`--no-predicted-reuse`): model learns reuse patterns implicitly

Models must be trained separately for each mode since the input dimensionality differs. Use `--no-predicted-reuse` consistently across generate, train, benchmark, and export steps.
