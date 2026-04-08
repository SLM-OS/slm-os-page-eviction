"""Feature normalization (Section 2.4 of the plan).

Applies feature-type-specific normalization to raw feature vectors
before model input. Handles ranks, counts, ticks, booleans,
categoricals, and priority values.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .extractor import FeatureConfig, PER_BLOCK_FEATURES, GLOBAL_FEATURES


@dataclass
class NormalizerStats:
    """Running statistics for adaptive normalization."""
    max_access_count: float = 1.0
    max_time_since_access: float = 1.0
    max_time_since_load: float = 1.0
    max_pending_loads: float = 1.0
    pool_size: int = 64


class FeatureNormalizer:
    """Normalizes raw feature vectors for model input.

    Normalization rules (from Section 2.4):
      - Ranks: / (N-1) -> [0, 1]
      - Counts: log1p then / log1p(max_observed)
      - Ticks: / horizon_window
      - Booleans: raw {0, 1}
      - Categoricals: ordinal for MLP, one-hot for XGBoost
      - Priority: / 7.0 -> [0, 1]
    """

    def __init__(
        self,
        config: FeatureConfig | None = None,
        stats: NormalizerStats | None = None,
    ):
        self.config = config or FeatureConfig()
        self.stats = stats or NormalizerStats()

    def normalize(
        self,
        features: np.ndarray,
        pool_size: int | None = None,
    ) -> np.ndarray:
        """Normalize a feature matrix (num_candidates x num_features).

        Args:
            features: Raw feature matrix from FeatureExtractor.
            pool_size: Current pool size for rank normalization.

        Returns:
            Normalized feature matrix with same shape.
        """
        if features.size == 0:
            return features

        normalized = features.copy()
        n = pool_size or self.stats.pool_size
        rank_denom = max(n - 1, 1)
        horizon = self.config.horizon_window

        # Build the column index mapping
        names = self.config.feature_names
        col = {name: i for i, name in enumerate(names)}

        # --- Per-block features ---

        # Ranks -> [0, 1]
        if "recency_rank" in col:
            normalized[:, col["recency_rank"]] /= rank_denom
        if "frequency_rank" in col:
            normalized[:, col["frequency_rank"]] /= rank_denom

        # Counts -> log1p normalized
        if "access_count" in col:
            raw = normalized[:, col["access_count"]]
            self.stats.max_access_count = max(
                self.stats.max_access_count, float(np.max(raw))
            )
            normalized[:, col["access_count"]] = (
                np.log1p(raw) / np.log1p(self.stats.max_access_count)
            )

        # Ticks -> / horizon
        if "time_since_access" in col:
            normalized[:, col["time_since_access"]] /= horizon
        if "time_since_load" in col:
            normalized[:, col["time_since_load"]] /= horizon

        # ref_count: leave as-is (small integer, bounded by MAX_TASKS)

        # Booleans: already {0, 1}
        # is_gpu_mapped, is_dirty: no normalization needed

        # pool_type: already {0, 1}

        # layer_idx_norm: already [0, 1] from extractor

        # model_priority -> / 7.0
        if "model_priority" in col:
            normalized[:, col["model_priority"]] /= self.config.max_priority

        # model_active_inferences: log1p normalize
        if "model_active_inferences" in col:
            raw = normalized[:, col["model_active_inferences"]]
            normalized[:, col["model_active_inferences"]] = np.log1p(raw) / np.log1p(10.0)

        # access_pattern: ordinal {0,1,2,3} / 3.0 for MLP
        if "access_pattern" in col:
            normalized[:, col["access_pattern"]] /= 3.0

        # predicted_reuse_dist: already [0, 1] from heuristic
        # eviction_cost: already [0, 1] from extractor

        # --- Global features ---

        # Utilizations: already [0, 1]
        # num_loaded_models: / max_models
        if "num_loaded_models" in col:
            normalized[:, col["num_loaded_models"]] /= self.config.max_models

        # total_gpu_mapped: / pool_size
        if "total_gpu_mapped" in col:
            normalized[:, col["total_gpu_mapped"]] /= max(n, 1)

        # pending_loads: log1p normalize
        if "pending_loads" in col:
            raw = normalized[:, col["pending_loads"]]
            self.stats.max_pending_loads = max(
                self.stats.max_pending_loads, float(np.max(raw)) if raw.size > 0 else 1.0
            )
            normalized[:, col["pending_loads"]] = (
                np.log1p(raw) / np.log1p(self.stats.max_pending_loads)
            )

        # avg_model_priority: / 7.0
        if "avg_model_priority" in col:
            normalized[:, col["avg_model_priority"]] /= self.config.max_priority

        # max_deadline_pressure: already [0, 1]
        # recent_fault_rate: already [0, 1]
        # hot_swap_active: already {0, 1}
        # req_block_pool: already {0, 1}

        # req_block_model_id: / max_models
        if "req_block_model_id" in col:
            normalized[:, col["req_block_model_id"]] /= self.config.max_models

        # req_block_priority: / 7.0
        if "req_block_priority" in col:
            normalized[:, col["req_block_priority"]] /= self.config.max_priority

        return normalized

    def to_onehot_categoricals(
        self,
        features: np.ndarray,
    ) -> np.ndarray:
        """Convert ordinal categoricals to one-hot for XGBoost.

        Expands pool_type (2 categories) and access_pattern (4 categories)
        into one-hot columns. Increases feature count by 4
        (replacing 2 ordinal columns with 2 + 4 one-hot columns).
        """
        names = self.config.feature_names
        col = {name: i for i, name in enumerate(names)}

        result_cols = []
        for i, name in enumerate(names):
            if name == "pool_type":
                # One-hot: WEIGHT=0, WORKSPACE=1
                pool_raw = features[:, i]
                result_cols.append((pool_raw == 0).astype(np.float32).reshape(-1, 1))
                result_cols.append((pool_raw >= 1).astype(np.float32).reshape(-1, 1))
            elif name == "access_pattern":
                # One-hot: SEQUENTIAL=0, RANDOM=1, STRIDED=2, BURST=3
                # Undo the /3.0 normalization first
                pattern_raw = np.round(features[:, i] * 3.0).astype(int)
                for cat in range(4):
                    result_cols.append(
                        (pattern_raw == cat).astype(np.float32).reshape(-1, 1)
                    )
            else:
                result_cols.append(features[:, i:i+1])

        return np.hstack(result_cols)
