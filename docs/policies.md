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
4. Decision and expert votes are recorded

Weight update (on feedback):
- **Fault** (evicted block re-accessed): multiply weight by `(1 - lr)` for experts that agreed
- **No fault**: multiply weight by `(1 + lr)` for experts that agreed
- Weights re-normalized after each update

Phase change detection: if weight distribution diverges significantly from uniform, a workload phase change is flagged.
