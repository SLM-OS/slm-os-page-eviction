# Architecture Overview

This document describes the architecture of the SLM-OS page replacement simulator and AI training pipeline.

## System Architecture

```
                    ┌─────────────────────────────────────────────────────┐
                    │                  Simulator Layer                     │
                    │                                                     │
 WorkloadGenerator ─┤  MemoryState ──── SimController ──── TraceCollector │
   7 scenarios      │   ├── weight_pool (Pool)     │          │          │
   6 model types    │   └── workspace_pool (Pool)  │          │          │
                    │       └── BlockMeta[]         │          │          │
                    │                               ▼          ▼          │
                    │                        EvictionPolicy  AccessEvent  │
                    │                        (select_victim)  EvictionEvent│
                    └─────────────────────────────────────────────────────┘
                                           │
                    ┌──────────────────────┼──────────────────────────────┐
                    │               Policy Layer                          │
                    │                      │                              │
                    │  ┌───────┐ ┌───────┐ ┌───────┐ ┌──────────────┐   │
                    │  │  LRU  │ │  LFU  │ │  ARC  │ │ SLM-Heuristic│   │
                    │  └───────┘ └───────┘ └───────┘ └──────────────┘   │
                    │  ┌───────┐ ┌───────┐ ┌─────────────────────────┐  │
                    │  │  MLP  │ │  XGB  │ │   CACHEUS (5 experts)   │  │
                    │  └───────┘ └───────┘ └─────────────────────────┘  │
                    │  ┌───────────────┐                                 │
                    │  │ Belady Oracle │  (offline optimal, labels only) │
                    │  └───────────────┘                                 │
                    └────────────────────────────────────────────────────┘
                                           │
                    ┌──────────────────────┼──────────────────────────────┐
                    │              Feature Layer                          │
                    │                      │                              │
                    │  FeatureExtractor ────┤  15 per-block features      │
                    │                      │  12 global state features    │
                    │                      │  = 27 total (26 w/o reuse)  │
                    │  FeatureNormalizer ───┤  rank / count / tick / bool │
                    └────────────────────────────────────────────────────┘
                                           │
                    ┌──────────────────────┼──────────────────────────────┐
                    │            Training Layer                           │
                    │                      │                              │
                    │  EvictionDataset ─── train/val/test split          │
                    │       │                                            │
                    │       ├── train_xgb.py ── XGBTrainConfig           │
                    │       ├── train_mlp.py ── PageReplacementMLP       │
                    │       ├── dagger.py ──── DAgger fine-tuning        │
                    │       └── evaluate.py ── PolicyEvaluator           │
                    └────────────────────────────────────────────────────┘
                                           │
                    ┌──────────────────────┼──────────────────────────────┐
                    │             Export Layer                            │
                    │                      │                              │
                    │  xgb_to_rust.py ─── JSON trees → Rust if-else      │
                    │  mlp_to_rust.py ─── weights → int8 const arrays    │
                    │  verify_export.py ── cross-validate Python vs Rust  │
                    └────────────────────────────────────────────────────┘
```

## Dual-Pool Memory Model

The simulator replicates SLM-OS's `ModelAllocator` with two fixed-size pools of 2MB blocks:

| Property | Weight Pool | Workspace Pool |
|----------|-------------|----------------|
| Default size | 128 MB (64 blocks) | 64 MB (32 blocks) |
| Permissions | Read-only | Read-write |
| Sharing | Multi-task (ref-counted) | Per-task |
| GPU-mappable | Yes | Yes |
| Typical content | Model layer weights | Activation buffers, KV cache |

Blocks are globally addressed with unique IDs across both pools. Workspace blocks get IDs offset by the weight pool size.

## Event Loop

The `SimController` processes one `AccessRequest` per tick:

1. **Hit check** -- scan pool for a block matching (model_id, layer_idx, pool_type)
2. **On hit** -- update access metadata, record access event, return
3. **On miss** -- increment fault counter
4. **Free block available** -- allocate into it, load data
5. **Pool full** -- ask `EvictionPolicy.select_victim()` for a victim index
6. **Evict** -- record eviction event, reset victim block, reuse it for the new load
7. **Feedback** -- call `policy.update_feedback()` for online learning policies

## Feature Vector (27 dimensions)

Each eviction candidate is represented by a 27-feature vector (or 26 without `predicted_reuse_dist`):

**Per-block (15):** recency_rank, frequency_rank, access_count, time_since_access, time_since_load, ref_count, is_gpu_mapped, pool_type, is_dirty, layer_idx_norm, model_priority, model_active_inferences, access_pattern, predicted_reuse_dist, eviction_cost

**Global (12):** weight_pool_util, workspace_pool_util, num_loaded_models, total_gpu_mapped, pending_loads, avg_model_priority, max_deadline_pressure, recent_fault_rate, hot_swap_active, req_block_pool, req_block_model_id, req_block_priority

The `predicted_reuse_dist` feature is controlled by `FeatureConfig(use_predicted_reuse=True/False)`. When disabled, models train on 26 features. Both variants are supported to compare whether the heuristic reuse estimate helps or introduces noise.

## CACHEUS Expert Selector

The `CACHEUSSelector` maintains a weighted ensemble of eviction policies:

1. Each expert scores all eviction candidates
2. Scores are combined using learned weights
3. The candidate with the highest weighted score is evicted
4. After eviction, if the evicted block is soon re-accessed (fault), experts that agreed with the bad decision are penalized; otherwise they are rewarded
5. Weights are normalized after each update

Phase change detection compares current weight distribution to uniform -- a large divergence indicates the selector has specialized to a particular workload pattern.

## Rust Export Pipeline

The export layer converts trained Python models to standalone Rust code:

**XGBoost:** Each tree is walked recursively, producing nested `if feature[i] < threshold { ... } else { ... }` chains. The final output applies a sigmoid to sum of tree outputs.

**MLP:** Weights are quantized to int8 with per-tensor symmetric scaling (`scale = max(|w|) / 127`). The generated Rust code contains `const` arrays for each layer's weights and biases, plus an inference function performing matrix multiplies with integer arithmetic and dequantization. Target model size: ~4KB (int8) or ~16KB (float32).
