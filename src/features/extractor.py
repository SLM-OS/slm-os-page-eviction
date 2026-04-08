"""Feature extractor: computes the 27-feature vector for eviction candidates.

Implements Sections 2.1-2.3 of the plan. For each eviction decision,
extracts 15 per-block features and 12 global features, producing a
27-dimensional vector per candidate.

The `use_predicted_reuse` flag controls whether feature #14
(predicted_reuse_dist) is included. When enabled, a heuristic estimate
of reuse distance is computed from access pattern and layer position.
When disabled, the feature is omitted and models are trained on 26
features, learning reuse patterns implicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from src.simulator.block import AccessPattern, BlockMeta, PoolType

if TYPE_CHECKING:
    from src.simulator.core import GlobalState, Pool


# Feature names for documentation and column headers
PER_BLOCK_FEATURES = [
    "recency_rank",
    "frequency_rank",
    "access_count",
    "time_since_access",
    "time_since_load",
    "ref_count",
    "is_gpu_mapped",
    "pool_type",
    "is_dirty",
    "layer_idx_norm",
    "model_priority",
    "model_active_inferences",
    "access_pattern",
    "predicted_reuse_dist",  # Optional: controlled by use_predicted_reuse
    "eviction_cost",
]

GLOBAL_FEATURES = [
    "weight_pool_util",
    "workspace_pool_util",
    "num_loaded_models",
    "total_gpu_mapped",
    "pending_loads",
    "avg_model_priority",
    "max_deadline_pressure",
    "recent_fault_rate",
    "hot_swap_active",
    "req_block_pool",
    "req_block_model_id",
    "req_block_priority",
]

NUM_PER_BLOCK_WITH_REUSE = 15
NUM_PER_BLOCK_WITHOUT_REUSE = 14
NUM_GLOBAL = 12


@dataclass
class FeatureConfig:
    """Configuration for feature extraction."""
    use_predicted_reuse: bool = True
    horizon_window: int = 1000  # Ticks for time normalization
    max_priority: int = 7       # Maximum priority level
    max_models: int = 8         # Maximum concurrent models

    @property
    def num_per_block(self) -> int:
        return NUM_PER_BLOCK_WITH_REUSE if self.use_predicted_reuse else NUM_PER_BLOCK_WITHOUT_REUSE

    @property
    def num_features(self) -> int:
        return self.num_per_block + NUM_GLOBAL

    @property
    def feature_names(self) -> list[str]:
        block_names = PER_BLOCK_FEATURES.copy()
        if not self.use_predicted_reuse:
            block_names.remove("predicted_reuse_dist")
        return block_names + GLOBAL_FEATURES


class FeatureExtractor:
    """Extracts feature vectors from block metadata and global state.

    Produces a (num_candidates, num_features) matrix for each eviction
    decision, suitable for input to XGBoost or MLP models.
    """

    def __init__(self, config: FeatureConfig | None = None):
        self.config = config or FeatureConfig()

    def extract_block_features(
        self,
        block: BlockMeta,
        pool: Pool,
        global_state: GlobalState,
        active_inferences: dict[int, int] | None = None,
    ) -> np.ndarray:
        """Extract per-block features for a single candidate.

        Returns a 1D array of length config.num_per_block.
        """
        active = active_inferences or {}
        num_allocated = len(pool.get_allocated_blocks())
        n = max(num_allocated - 1, 1)  # For rank normalization

        # Determine total layers for layer index normalization
        # Use model's weight blocks as a proxy for total layers
        total_layers = max(1, num_allocated)

        features = [
            pool.recency_rank(block),                           # recency_rank
            pool.frequency_rank(block),                         # frequency_rank
            block.access_count,                                 # access_count (raw, normalized later)
            global_state.tick - block.last_access_time,         # time_since_access
            global_state.tick - block.load_time,                # time_since_load
            block.ref_count,                                    # ref_count
            int(block.gpu_mapped),                              # is_gpu_mapped
            int(block.pool_type),                               # pool_type
            int(block.is_dirty),                                # is_dirty
            self._normalize_layer_idx(block),                   # layer_idx_norm
            block.model_id,                                     # model_priority (resolved later from config)
            active.get(block.model_id, 0),                      # model_active_inferences
            int(block.access_pattern),                          # access_pattern
        ]

        if self.config.use_predicted_reuse:
            features.append(self._predict_reuse_heuristic(block, global_state))

        features.append(self._compute_eviction_cost(block))     # eviction_cost

        return np.array(features, dtype=np.float32)

    def extract_global_features(self, global_state: GlobalState) -> np.ndarray:
        """Extract global state features.

        Returns a 1D array of length NUM_GLOBAL (12).
        """
        return np.array([
            global_state.weight_pool_utilization,
            global_state.workspace_pool_utilization,
            global_state.num_loaded_models,
            global_state.total_gpu_mapped_blocks,
            global_state.pending_load_requests,
            global_state.avg_model_priority,
            global_state.max_deadline_pressure,
            global_state.recent_fault_rate,
            int(global_state.hot_swap_in_progress),
            int(global_state.requesting_block_pool),
            global_state.requesting_block_model_id,
            global_state.requesting_block_priority,
        ], dtype=np.float32)

    def extract_candidate_features(
        self,
        candidates: list[BlockMeta],
        pool: Pool,
        global_state: GlobalState,
        active_inferences: dict[int, int] | None = None,
    ) -> np.ndarray:
        """Extract features for all eviction candidates.

        Returns a (num_candidates, num_features) matrix where each row
        is the concatenation of per-block features and global features.
        """
        global_feats = self.extract_global_features(global_state)
        rows = []
        for block in candidates:
            block_feats = self.extract_block_features(
                block, pool, global_state, active_inferences
            )
            row = np.concatenate([block_feats, global_feats])
            rows.append(row)
        return np.stack(rows) if rows else np.empty((0, self.config.num_features))

    def _normalize_layer_idx(self, block: BlockMeta) -> float:
        """Normalize layer index to [0, 1]. Workspace blocks map to 0."""
        if block.layer_idx < 0:
            return 0.0
        # Approximate normalization using a reasonable max
        # (exact normalization happens in FeatureNormalizer)
        return min(block.layer_idx / 32.0, 1.0)

    def _predict_reuse_heuristic(
        self, block: BlockMeta, global_state: GlobalState
    ) -> float:
        """Heuristic estimate of reuse distance (feature #14).

        Combines access pattern type and layer position to estimate when
        the block will next be accessed. This is NOT the oracle value --
        it is the same heuristic used at both training and inference time.

        Returns a value in [0.0, 1.0] where 0.0 = imminent reuse,
        1.0 = unlikely to be reused soon.
        """
        pattern = block.access_pattern
        layer_norm = self._normalize_layer_idx(block)
        time_since = global_state.tick - block.last_access_time
        horizon = self.config.horizon_window

        if pattern == AccessPattern.SEQUENTIAL:
            # Sequential sweeps: blocks near the current position will
            # be reused soon; blocks far behind in the sweep are distant.
            return min(time_since / horizon, 1.0)
        elif pattern == AccessPattern.BURST:
            # Burst access: recently burst blocks likely to burst again
            recency = min(time_since / (horizon * 0.1), 1.0)
            return recency * 0.5  # Low reuse distance for burst patterns
        elif pattern == AccessPattern.STRIDED:
            # Strided: depends on position relative to stride pattern
            return min(time_since / (horizon * 0.5), 1.0)
        else:
            # Random: no predictable reuse
            return 0.8

    @staticmethod
    def _compute_eviction_cost(block: BlockMeta) -> float:
        """Normalized eviction cost accounting for writeback and GPU unmap."""
        cost = 0.0
        if block.is_dirty:
            cost += 0.3
        if block.gpu_mapped:
            cost += 0.5
        # Workspace is cheaper to recreate than weights
        if block.pool_type == PoolType.WORKSPACE:
            cost *= 0.5
        return min(cost + 0.2, 1.0)  # Base cost + extras, clamped to [0, 1]
