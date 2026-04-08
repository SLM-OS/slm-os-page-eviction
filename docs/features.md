# Feature Engineering

The feature vector encodes what the eviction policy "sees" at each decision point. Code lives in `src/features/`.

## Feature Vector Layout

Each eviction candidate is represented by a flat vector of 27 features (or 26 without `predicted_reuse_dist`). The vector is the concatenation of 15 per-block features and 12 global context features.

### Per-Block Features (15)

Computed for each eviction candidate from its `BlockMeta` and the pool state:

| # | Name | Type | Range | Description |
|---|------|------|-------|-------------|
| 1 | `recency_rank` | int | [0, N-1] | Rank by last access time (0 = most recent) |
| 2 | `frequency_rank` | int | [0, N-1] | Rank by access count (0 = most accessed) |
| 3 | `access_count` | int | [0, inf) | Total accesses since load |
| 4 | `time_since_access` | int | [0, inf) | Ticks since last access |
| 5 | `time_since_load` | int | [0, inf) | Ticks since block was loaded |
| 6 | `ref_count` | int | [0, MAX] | Active task references (0 = evictable) |
| 7 | `is_gpu_mapped` | bool | {0, 1} | Currently mapped to GPU |
| 8 | `pool_type` | cat | {0, 1} | 0=WEIGHT, 1=WORKSPACE |
| 9 | `is_dirty` | bool | {0, 1} | Modified since load (writeback cost) |
| 10 | `layer_idx_norm` | float | [0, 1] | Normalized layer position (-1 maps to 0) |
| 11 | `model_priority` | int | [0, 7] | Owning model's priority level |
| 12 | `model_active_inferences` | int | [0, inf) | Active inference tasks for this model |
| 13 | `access_pattern` | cat | {0,1,2,3} | SEQUENTIAL/RANDOM/STRIDED/BURST |
| 14 | `predicted_reuse_dist` | float | [0, 1] | Heuristic reuse distance estimate (optional) |
| 15 | `eviction_cost` | float | [0, 1] | Normalized cost to reload |

### Global Features (12)

Computed once per eviction decision from `GlobalState`:

| # | Name | Type | Range | Description |
|---|------|------|-------|-------------|
| 16 | `weight_pool_util` | float | [0, 1] | Weight pool utilization |
| 17 | `workspace_pool_util` | float | [0, 1] | Workspace pool utilization |
| 18 | `num_loaded_models` | int | [0, 8] | Distinct loaded models |
| 19 | `total_gpu_mapped` | int | [0, N] | GPU-mapped block count |
| 20 | `pending_loads` | int | [0, inf) | Queued load requests |
| 21 | `avg_model_priority` | float | [0, 7] | Mean priority of loaded models |
| 22 | `max_deadline_pressure` | float | [0, 1] | Highest deadline urgency |
| 23 | `recent_fault_rate` | float | [0, 1] | Faults/accesses in last 100 ticks |
| 24 | `hot_swap_active` | bool | {0, 1} | Model swap in progress |
| 25 | `req_block_pool` | cat | {0, 1} | Pool type of incoming block |
| 26 | `req_block_model_id` | int | [0, 8] | Model ID of incoming block |
| 27 | `req_block_priority` | int | [0, 7] | Priority of incoming block's model |

## Predicted Reuse Distance (Feature #14)

This is a **heuristic estimate**, not the oracle value. It combines access pattern type and recency to predict when a block will next be accessed:

- **SEQUENTIAL**: `min(time_since_access / horizon, 1.0)` -- blocks far behind in a sweep are distant
- **BURST**: `min(time_since / (0.1 * horizon), 1.0) * 0.5` -- burst blocks likely reused soon
- **STRIDED**: `min(time_since / (0.5 * horizon), 1.0)` -- moderate prediction
- **RANDOM**: constant `0.8` -- no predictable reuse

This feature is **optional**. It is controlled by `FeatureConfig(use_predicted_reuse=True/False)`:
- `True` (default): 27 features, includes the heuristic estimate
- `False`: 26 features, the model learns reuse patterns implicitly

Since this changes the feature vector dimensionality, models must be trained separately for each variant. The dual-mode support lets us empirically measure whether the heuristic helps.

## Normalization (`normalizer.py`)

`FeatureNormalizer` applies type-specific normalization:

| Feature Group | Method | Notes |
|---------------|--------|-------|
| Ranks | `/ (pool_size - 1)` | Maps to [0, 1] |
| Counts | `log1p(x) / log1p(max)` | Compresses heavy tails |
| Ticks | `/ horizon_window` | Default horizon = 1000 |
| Booleans | Raw {0, 1} | No transformation |
| Categoricals | Ordinal for MLP, one-hot for XGBoost | `to_onehot_categoricals()` helper |
| Priority | `/ 7.0` | Maps 8 levels to [0, 1] |

## Usage

```python
from src.features.extractor import FeatureExtractor, FeatureConfig
from src.features.normalizer import FeatureNormalizer

config = FeatureConfig(use_predicted_reuse=True)
extractor = FeatureExtractor(config)
normalizer = FeatureNormalizer(config)

# Extract features for all candidates at an eviction point
features = extractor.extract_candidate_features(candidates, pool, global_state)
# features.shape == (num_candidates, 27)

# Normalize for model input
features_norm = normalizer.normalize(features, pool_size=pool.num_blocks)
```
