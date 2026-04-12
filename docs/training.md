# Training Pipeline

The training pipeline converts simulator traces into trained eviction models. Code lives in `src/training/` with CLI entry points in `scripts/`.

## End-to-End Flow

```
generate_dataset.py           train_all.py              benchmark.py         export_to_slmos.py
       │                           │                          │                       │
  7 scenarios x N seeds     Load Parquet             Run all policies         Load trained models
       │                    train/val/test split      on all scenarios         Export to Rust
  LRU trace → Belady oracle      │                   Produce comparison       Verify predictions
       │                    Train XGBoost              table + summary
  _LabelingPolicy captures  Train MLP                      │
  features at each eviction  (+ optional DAgger)     Save CSV results
       │                         │
  Write eviction_events.parquet  Save models to data/models/
```

## Dataset Generation (`scripts/generate_dataset.py`)

Runs the simulator for each scenario/seed combination, using a `_LabelingPolicy` wrapper that:
1. Delegates eviction decisions to the Belady oracle
2. At each eviction, extracts features for all candidates using `FeatureExtractor`
3. Labels the Belady-optimal candidate (`is_optimal=1`, others `is_optimal=0`)
4. Records reuse distances for the regression target

The approach captures features **in-situ** during simulation (not post-hoc), ensuring block metadata is current at the time of feature extraction.

```bash
# Full dataset (7 scenarios x 5 seeds, default 64/32 pool sizes)
python scripts/generate_dataset.py

# Quick test (1 seed, small pools)
python scripts/generate_dataset.py --seeds 1 --weight-blocks 16 --workspace-blocks 8

# Without predicted_reuse_dist feature (26 features)
python scripts/generate_dataset.py --no-predicted-reuse
```

Output: `data/traces/eviction_events.parquet`

### Dataset Schema

Each row is one eviction candidate at one eviction decision point:

| Column | Type | Description |
|--------|------|-------------|
| 27 feature columns | float32 | Feature vector (see `FeatureConfig.feature_names`) |
| `is_optimal` | int8 | 1 if Belady chose this candidate, 0 otherwise |
| `reuse_distance` | float32 | Normalized ticks until next access (regression target) |
| `eviction_id` | int64 | Groups candidates for the same eviction decision |
| `scenario` | string | Workload scenario name |

### Dataset Split (`dataset.py`)

- **Test set**: all rows from `hot_swap` and `gpu_contention` scenarios (held-out workloads)
- **Train/val**: remaining scenarios, split by `eviction_id` (not by row) to keep candidate groups together
- Default ratio: 70% train, 15% val, 15% test

## XGBoost Training (`train_xgb.py`)

```python
from src.training.train_xgb import train_xgboost, XGBTrainConfig, cross_validate_xgb

config = XGBTrainConfig(
    max_depth=6,
    num_rounds=200,
    learning_rate=0.1,
    scale_pos_weight=30.0,  # Auto-adjusted from actual class balance
    early_stopping_rounds=10,
)

result = train_xgboost(train_data, val_data, config)
# result.model: xgb.Booster
# result.val_auc: float
# result.feature_importance: dict[str, float]
```

Key features:
- Dynamic `scale_pos_weight` calculation from actual class balance
- 5-fold cross-validation by scenario (not by row)
- Feature importance extraction (gain-based)
- Early stopping on validation AUC

### Hyperparameter Grid Search

```python
from src.training.train_xgb import grid_search_xgb

results = grid_search_xgb(train_data, val_data, param_grid={
    "max_depth": [4, 5, 6, 7],
    "learning_rate": [0.05, 0.1, 0.2],
    "min_child_weight": [5, 10, 20],
})
# results is a list of {**params, "val_auc": float, "best_iteration": int}
# sorted by val_auc descending
```

The default grid covers 36 configurations. Empirically, the task is well-separated (all configs achieve > 0.9996 AUC); the default depth=6 / lr=0.1 is within 0.002% of the best (depth=7 / lr=0.2) while producing a smaller Rust export.

## MLP Training (`train_mlp.py`)

```python
from src.training.train_mlp import train_mlp, MLPTrainConfig, PageReplacementMLP

config = MLPTrainConfig(
    hidden_sizes=[64, 32, 16],
    learning_rate=1e-3,
    batch_size=256,
    max_epochs=50,
    patience=5,
    reuse_loss_weight=0.3,
)

result = train_mlp(train_data, val_data, config)
# result.model: PageReplacementMLP
# result.best_val_loss: float
# result.best_epoch: int
```

### Model Architecture

```
Input (27) → Linear(64) → ReLU → Dropout(0.1)
          → Linear(32) → ReLU → Dropout(0.1)
          → Linear(16) → ReLU
          ├→ classify_head: Linear(1) → Sigmoid  (is_optimal)
          └→ reuse_head: Linear(1) → ReLU        (reuse_distance)
```

Joint loss: `L = BCE(is_optimal) + 0.3 * MSE(reuse_distance)`

The reuse distance head regularizes the shared representation, encouraging the model to learn about future access patterns rather than memorizing specific eviction decisions.

## DAgger Fine-Tuning (`dagger.py`)

Dataset Aggregation closes the distribution shift between offline training data (collected under Belady's policy) and the MLP's own behavior:

1. Run the MLP policy in the simulator
2. At each eviction the MLP encounters, also compute the Belady-optimal label
3. Collect `(features, optimal_label)` pairs from the MLP's own state distribution
4. Merge new data with original training set
5. Retrain the MLP
6. Repeat for N rounds (default: 3)

```python
from src.training.dagger import dagger_finetune, DAggerConfig

result = dagger_finetune(
    mlp_policy, oracle, train_data, val_data,
    scenario_requests, memory,
    config=DAggerConfig(num_rounds=3, mix_ratio=0.5),
)
```

## Policy Evaluation (`evaluate.py`)

`PolicyEvaluator` runs policies in the simulator and collects metrics:

```python
evaluator = PolicyEvaluator(weight_blocks=64, workspace_blocks=32)
results_df = evaluator.run_benchmark(policies, scenarios, seeds)
summary = evaluator.summarize(results_df)
```

The benchmark computes normalized fault rates using LRU as the baseline (1.0) and Belady as optimal (0.0). Results are returned as a DataFrame with columns: policy, scenario, seed, fault_rate, normalized_fault_rate, total_evictions.

## CLI Scripts

### `scripts/train_all.py`

Trains both XGBoost and MLP models from a generated dataset:

```bash
python scripts/train_all.py --data data/traces/eviction_events.parquet --output data/models/
```

### `scripts/benchmark.py`

Runs all policies across all scenarios. Automatically loads trained models from `--model-dir` if available:

```bash
python scripts/benchmark.py --output-dir data/results/ --model-dir data/models/ --seeds 5
```

### `scripts/export_to_slmos.py`

Exports trained models as Rust source code:

```bash
python scripts/export_to_slmos.py --model-dir data/models/ --output-dir data/export/
```

### `scripts/analyze_models.py`

Analyzes trained models on the test set: feature importance, per-scenario eviction decision accuracy, and int8 quantization verification:

```bash
python scripts/analyze_models.py --data data/traces/eviction_events.parquet --model-dir data/models/
```

Prints gain-ranked feature table (with % of total gain), per-scenario test accuracy for each model, and a quantization PASS/FAIL against the 99% decision-agreement target.

### `scripts/feature_reduction.py`

Trains XGBoost on the top-10 features only (identified from gain ranking) and profiles per-candidate inference latency for XGBoost and MLP at different batch sizes:

```bash
python scripts/feature_reduction.py --data data/traces/eviction_events.parquet --model-dir data/models/
```

### `scripts/run_dagger.py`

Runs DAgger fine-tuning on an existing MLP checkpoint and compares its test-set accuracy against the baseline:

```bash
python scripts/run_dagger.py --data data/traces/eviction_events.parquet --model-dir data/models/ --dagger-rounds 3
```

Saves the fine-tuned model to `data/models/mlp_model_dagger.pt`. Note: when the baseline MLP is already near-ceiling, DAgger may not improve accuracy — this is a valid finding, not a bug.

### `scripts/tune_cacheus.py`

Phase 5 hyperparameter sweep for the CACHEUS expert selector: tries a grid of `(learning_rate, window_size)` and four expert-pool compositions (`classical_only`, `ml_only`, `ml_plus_lru`, `all_5`), then records weight trajectories on each scenario for the best config:

```bash
python scripts/tune_cacheus.py --model-dir data/models/ --output-dir data/results/ --seeds 3
```

Outputs:
- `cacheus_hyperparam_sweep.csv` — mean/max norm fault rate per (lr, window)
- `cacheus_pool_comparison.csv` — per-scenario norm rate per pool composition
- `cacheus_trajectories.json` — weight-over-time data for each scenario

### `scripts/analyze_benchmark.py`

Phase 7 statistical analysis of `benchmark_results.csv`:

```bash
python scripts/analyze_benchmark.py --input data/results/benchmark_results.csv --output-dir data/results/
```

Produces:
- `policy_scenario_matrix.csv` — pivot table of mean normalized fault rate
- `pairwise_ttests.csv` — per-scenario paired t-tests for every policy pair
- `failure_analysis.csv` — scenarios where each policy exceeds 0.5 norm rate
- `policy_comparison_per_scenario.png`, `policy_overall_ranking.png`, `policy_heatmap.png`

### `scripts/plot_trajectories.py`

Renders CACHEUS weight trajectories from `cacheus_trajectories.json`:

```bash
python scripts/plot_trajectories.py --input data/results/cacheus_trajectories.json --output-dir data/results/
```

Produces per-scenario PNGs plus a combined grid, and prints adaptation-speed estimates (ticks until weight change drops below threshold).
