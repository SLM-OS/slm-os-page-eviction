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
