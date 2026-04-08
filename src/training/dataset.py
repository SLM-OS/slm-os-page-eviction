"""Dataset loading, splitting, and management for training.

Handles Parquet-based datasets with Belady-optimal labels. Implements
the train/val/test split strategy from Section 6.4: split by eviction_id
(not by row) to keep candidate sets together, with held-out scenarios
for the test set.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.extractor import FeatureConfig


# Held-out scenarios for the test set (unseen workloads)
TEST_SCENARIOS = {"hot_swap", "gpu_contention"}


@dataclass
class EvictionDataset:
    """A labeled dataset of eviction decision points.

    Each row represents one eviction candidate with its features and labels.
    Rows are grouped by eviction_id (all candidates for a single decision).
    """
    features: np.ndarray        # (num_rows, num_features)
    is_optimal: np.ndarray      # (num_rows,) binary label
    reuse_distance: np.ndarray  # (num_rows,) regression target
    eviction_ids: np.ndarray    # (num_rows,) group key
    scenarios: np.ndarray       # (num_rows,) scenario name
    feature_names: list[str]

    @property
    def num_rows(self) -> int:
        return self.features.shape[0]

    @property
    def num_features(self) -> int:
        return self.features.shape[1]

    @property
    def num_evictions(self) -> int:
        return len(np.unique(self.eviction_ids))

    @property
    def class_balance(self) -> float:
        """Fraction of positive (is_optimal=1) labels."""
        return float(np.mean(self.is_optimal))

    def get_eviction_group(self, eviction_id: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (features, labels) for a single eviction event."""
        mask = self.eviction_ids == eviction_id
        return self.features[mask], self.is_optimal[mask]


def load_dataset(
    path: str | Path,
    config: FeatureConfig | None = None,
) -> EvictionDataset:
    """Load a labeled dataset from Parquet.

    Args:
        path: Path to the Parquet file matching schema from Section 6.3.
        config: Feature configuration (determines which columns to use).

    Returns:
        An EvictionDataset with features and labels.
    """
    cfg = config or FeatureConfig()
    df = pd.read_parquet(path)

    feature_cols = cfg.feature_names
    features = df[feature_cols].values.astype(np.float32)
    is_optimal = df["is_optimal"].values.astype(np.int8)
    reuse_distance = df["reuse_distance"].values.astype(np.float32)
    eviction_ids = df["eviction_id"].values.astype(np.int64)
    scenarios = df["scenario"].values

    return EvictionDataset(
        features=features,
        is_optimal=is_optimal,
        reuse_distance=reuse_distance,
        eviction_ids=eviction_ids,
        scenarios=scenarios,
        feature_names=feature_cols,
    )


def train_val_test_split(
    dataset: EvictionDataset,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 42,
) -> tuple[EvictionDataset, EvictionDataset, EvictionDataset]:
    """Split dataset into train/validation/test sets.

    Strategy (Section 6.4):
      - Test set: all rows from held-out scenarios (hot_swap, gpu_contention)
      - Train/val: remaining rows, split by eviction_id (not by row)
        to keep candidate sets together
    """
    rng = np.random.default_rng(seed)

    # Separate test scenarios
    test_mask = np.isin(dataset.scenarios, list(TEST_SCENARIOS))
    trainval_mask = ~test_mask

    # Split train/val by eviction_id
    trainval_eviction_ids = np.unique(dataset.eviction_ids[trainval_mask])
    rng.shuffle(trainval_eviction_ids)

    adjusted_train_frac = train_frac / (train_frac + val_frac)
    split_idx = int(len(trainval_eviction_ids) * adjusted_train_frac)
    train_ids = set(trainval_eviction_ids[:split_idx])
    val_ids = set(trainval_eviction_ids[split_idx:])

    train_mask = np.array([
        (not test_mask[i]) and (dataset.eviction_ids[i] in train_ids)
        for i in range(dataset.num_rows)
    ])
    val_mask = np.array([
        (not test_mask[i]) and (dataset.eviction_ids[i] in val_ids)
        for i in range(dataset.num_rows)
    ])

    def _subset(mask: np.ndarray) -> EvictionDataset:
        return EvictionDataset(
            features=dataset.features[mask],
            is_optimal=dataset.is_optimal[mask],
            reuse_distance=dataset.reuse_distance[mask],
            eviction_ids=dataset.eviction_ids[mask],
            scenarios=dataset.scenarios[mask],
            feature_names=dataset.feature_names,
        )

    return _subset(train_mask), _subset(val_mask), _subset(test_mask)
