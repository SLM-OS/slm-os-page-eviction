# Eviction Policies

All policies implement the `EvictionPolicy` ABC defined in `src/policies/base.py`.

## Interface

```python
class EvictionPolicy(ABC):
    def select_victim(self, candidates: list[BlockMeta], pool: Pool, global_state: GlobalState) -> int:
        """Return index into candidates list of the block to evict."""

    def update_feedback(self, block_id: int, was_fault: bool) -> None:
        """Optional: called after eviction for online learning."""

    def score(self, candidates: list[BlockMeta], pool: Pool, global_state: GlobalState) -> list[float]:
        """Score each candidate (higher = more evictable). Used by CACHEUS."""

    def reset(self) -> None:
        """Reset internal state for a new simulation run."""

    def name(self) -> str:
        """Human-readable policy name."""
```

## Classical Policies

### LRU (`lru.py`)

Evicts the block with the oldest `last_access_time`. Good for sequential weight sweeps. Serves as the normalization baseline (normalized_fault_rate = 1.0).

### LFU (`lfu.py`)

Evicts the block with the lowest `access_count`. Ties broken by LRU (oldest among least-frequently-used). Good for stable hot sets.

### ARC (`arc.py`)

Adaptive Replacement Cache. Maintains four lists:
- **T1** -- recently accessed once (recency)
- **T2** -- recently accessed more than once (frequency)
- **B1** -- ghost list of recently evicted T1 entries
- **B2** -- ghost list of recently evicted T2 entries

A hit in B1 increases T1's share (more recency); a hit in B2 increases T2's share (more frequency). This adapts the recency/frequency balance to the workload.

### SLM-Heuristic (`slm_heuristic.py`)

Hand-tuned policy matching the current SLM-OS eviction logic:
1. Prefer evicting workspace blocks (cheaper to recreate than weights)
2. Among workspace, evict LRU
3. If no workspace evictable, prefer weights from inactive models
4. Fallback: LRU among remaining

### Belady Oracle (`belady.py`)

Offline optimal. Given complete future access trace, evicts the block whose next access is furthest in the future. Ties broken by eviction cost (prefer cheaper blocks).

Used for:
- **Training label generation** -- `label_eviction()` returns optimal index + reuse distances
- **Upper bound** -- normalized_fault_rate = 0.0 by definition
- **Reuse distance computation** -- `get_reuse_distance()` for auxiliary regression target

## ML Policies

### XGBPolicy (`xgb_policy.py`)

Wraps a trained `xgboost.Booster`. For each eviction:
1. Extract 27-feature vectors for all candidates via `FeatureExtractor`
2. Normalize via `FeatureNormalizer`
3. Create `DMatrix`, call `booster.predict()`
4. Evict candidate with highest probability

### MLPPolicy (`mlp_policy.py`)

Wraps a trained `PageReplacementMLP`. Same flow as XGBPolicy but uses PyTorch inference:
1. Extract and normalize features
2. Convert to `torch.Tensor`
3. Call `model.predict_scores()` (forward pass, returns sigmoid probabilities)
4. Evict candidate with highest probability

Exposes `feature_config` property and `update_model()` for DAgger fine-tuning.

## CACHEUS Selector (`cacheus.py`)

Weighted ensemble of up to 5 experts. At each eviction:
1. Each expert produces scores via `score()`
2. Scores are combined: `weighted_score[i] = sum(w[k] * expert_k_score[i])`
3. Candidate with highest weighted score is evicted
4. Decision and expert votes (per-expert chosen index plus the ensemble's actual choice) are recorded in `EvictionRecord`

### Feedback-Driven Weight Update

The simulator drives CACHEUS feedback by tracking recently evicted content. When content `(model_id, layer_idx, pool_type)` is evicted, the simulator records it in `_evicted_content`. Later:

- **Re-access (bad eviction)**: when a miss occurs for content that was recently evicted, `update_feedback(block_id, was_fault=True)` fires
- **Window expiry (good eviction)**: content not re-accessed within `_eviction_feedback_window` ticks yields `update_feedback(block_id, was_fault=False)`

Weight update rule (on `update_feedback`):

| Eviction outcome | Expert agreed with ensemble | Expert disagreed with ensemble |
|---|---|---|
| Bad (`was_fault=True`) | `w *= (1 - lr)` (penalize) | `w *= (1 + 0.5 * lr)` (small reward) |
| Good (`was_fault=False`) | `w *= (1 + lr)` (reward) | unchanged |

Weights are floored at `min_weight` then renormalized so experts stay recoverable even after long penalty runs.

**Agreement is measured against `EvictionRecord.ensemble_choice`** — the candidate index the combined scores actually selected — not any single expert's choice. (Earlier drafts incorrectly compared against the last expert; fixed during Phase 3-4 evaluation.)

### Phase Change Detection

`detect_phase_change(threshold)` flags a workload phase transition when the maximum per-expert deviation from the uniform distribution exceeds `threshold`. The detector requires at least `window_size / 2` recorded decisions before returning True to avoid noise.
