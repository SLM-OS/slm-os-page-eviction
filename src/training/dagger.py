"""DAgger (Dataset Aggregation) fine-tuning for the MLP (Section 8.2 Phase 4).

Runs the MLP policy in the simulator, collects on-policy data with
Belady-optimal labels, augments the training set, and retrains.
This closes the distribution shift between offline training data
(collected under Belady/LRU policies) and the MLP's own behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .dataset import EvictionDataset
from .train_mlp import MLPTrainConfig, MLPTrainResult, train_mlp

if TYPE_CHECKING:
    from src.features.extractor import FeatureExtractor
    from src.policies.belady import BeladyOracle
    from src.policies.mlp_policy import MLPPolicy
    from src.simulator.core import MemoryState, SimController
    from src.simulator.trace import TraceCollector
    from src.simulator.workload import AccessRequest


@dataclass
class DAggerConfig:
    """DAgger fine-tuning configuration."""
    num_rounds: int = 3
    mlp_config: MLPTrainConfig | None = None
    mix_ratio: float = 0.5  # Fraction of new data vs original per round


@dataclass
class DAggerResult:
    """Results from DAgger fine-tuning."""
    round_results: list[MLPTrainResult]
    final_dataset_size: int
    improvement_per_round: list[float]  # Validation loss per round


def dagger_finetune(
    mlp_policy: MLPPolicy,
    oracle: BeladyOracle,
    original_train: EvictionDataset,
    val_data: EvictionDataset,
    scenario_requests: list[AccessRequest],
    memory: MemoryState,
    config: DAggerConfig | None = None,
) -> DAggerResult:
    """Run DAgger fine-tuning loop.

    For each round:
      1. Run the current MLP policy in the simulator
      2. At each eviction, also compute the Belady-optimal decision
      3. Collect (state, optimal_label) pairs from the MLP's visited states
      4. Augment the training set with these new samples
      5. Retrain the MLP on the augmented dataset

    Args:
        mlp_policy: Current MLP policy to improve.
        oracle: Belady oracle with access to future trace.
        original_train: Original offline training dataset.
        val_data: Validation dataset for evaluation.
        scenario_requests: Access requests to run the MLP policy on.
        memory: Memory state for the simulator.
        config: DAgger configuration.

    Returns:
        DAggerResult with per-round training results.
    """
    cfg = config or DAggerConfig()
    mlp_cfg = cfg.mlp_config or MLPTrainConfig()

    round_results: list[MLPTrainResult] = []
    improvements: list[float] = []
    augmented_train = original_train

    for round_idx in range(cfg.num_rounds):
        # Step 1-3: Run MLP in simulator, collect on-policy data with oracle labels
        new_data = _collect_on_policy_data(
            mlp_policy, oracle, scenario_requests, memory
        )

        if new_data is not None:
            # Step 4: Augment training set
            augmented_train = _merge_datasets(
                augmented_train, new_data, cfg.mix_ratio
            )

        # Step 5: Retrain
        result = train_mlp(augmented_train, val_data, mlp_cfg)
        round_results.append(result)
        improvements.append(result.best_val_loss)

        # Update the MLP policy with the new model
        mlp_policy.update_model(result.model)

    return DAggerResult(
        round_results=round_results,
        final_dataset_size=augmented_train.num_rows,
        improvement_per_round=improvements,
    )


class _DAggerLabelingPolicy:
    """MLP drives eviction decisions; Belady provides ground-truth labels.

    Not a real EvictionPolicy subclass — it wraps an MLP policy and records
    features with oracle labels at each eviction the MLP encounters.
    """

    def __init__(
        self,
        mlp_policy: MLPPolicy,
        oracle: BeladyOracle,
        extractor: FeatureExtractor,
    ):
        self.mlp_policy = mlp_policy
        self.oracle = oracle
        self.extractor = extractor
        self.rows_features: list[np.ndarray] = []
        self.rows_is_optimal: list[int] = []
        self.rows_reuse_dist: list[float] = []
        self.rows_eviction_id: list[int] = []
        self._eviction_id = 0

    def select_victim(self, candidates, pool, global_state):
        # Belady labels the optimal choice
        optimal_idx, reuse_distances = self.oracle.label_eviction(
            candidates, pool, global_state
        )
        # Extract features while block state is current
        features = self.extractor.extract_candidate_features(
            candidates, pool, global_state
        )
        for i in range(len(candidates)):
            self.rows_features.append(features[i])
            self.rows_is_optimal.append(1 if i == optimal_idx else 0)
            self.rows_reuse_dist.append(reuse_distances[i])
            self.rows_eviction_id.append(self._eviction_id)
        self._eviction_id += 1

        # MLP makes the actual decision (exploring its own distribution)
        return self.mlp_policy.select_victim(candidates, pool, global_state)

    def update_feedback(self, block_id, was_fault):
        self.mlp_policy.update_feedback(block_id, was_fault)

    def reset(self):
        self.mlp_policy.reset()

    def name(self):
        return "DAgger-MLP"

    def score(self, candidates, pool, global_state):
        return self.mlp_policy.score(candidates, pool, global_state)


def _collect_on_policy_data(
    mlp_policy: MLPPolicy,
    oracle: BeladyOracle,
    requests: list[AccessRequest],
    memory: MemoryState,
) -> EvictionDataset | None:
    """Run MLP policy and collect oracle-labeled data from visited states.

    This is where DAgger differs from standard imitation learning:
    the data comes from the learner's own state distribution, not the
    expert's. The labels still come from the expert (Belady oracle).
    """
    from src.features.extractor import FeatureExtractor
    from src.simulator.core import SimController
    from src.simulator.trace import TraceCollector

    extractor = FeatureExtractor(mlp_policy.feature_config)
    labeler = _DAggerLabelingPolicy(mlp_policy, oracle, extractor)

    # Run with a fresh memory state so we don't mutate the caller's
    fresh_memory = MemoryState(
        memory.weight_pool.num_blocks,
        memory.workspace_pool.num_blocks,
    )
    trace = TraceCollector()
    sim = SimController(memory=fresh_memory, policy=labeler, trace_collector=trace)
    sim.run(requests)

    if not labeler.rows_features:
        return None

    return EvictionDataset(
        features=np.stack(labeler.rows_features),
        is_optimal=np.array(labeler.rows_is_optimal, dtype=np.int8),
        reuse_distance=np.array(labeler.rows_reuse_dist, dtype=np.float32),
        eviction_ids=np.array(labeler.rows_eviction_id, dtype=np.int64),
        scenarios=np.array(["dagger"] * len(labeler.rows_features)),
        feature_names=extractor.config.feature_names,
    )


def _merge_datasets(
    original: EvictionDataset,
    new_data: EvictionDataset,
    mix_ratio: float,
) -> EvictionDataset:
    """Merge original and new data with the specified mixing ratio."""
    # Sample from new data according to mix_ratio
    n_new = int(new_data.num_rows * mix_ratio)
    if n_new == 0:
        return original

    rng = np.random.default_rng(42)
    indices = rng.choice(new_data.num_rows, size=min(n_new, new_data.num_rows), replace=False)

    return EvictionDataset(
        features=np.concatenate([original.features, new_data.features[indices]]),
        is_optimal=np.concatenate([original.is_optimal, new_data.is_optimal[indices]]),
        reuse_distance=np.concatenate([original.reuse_distance, new_data.reuse_distance[indices]]),
        eviction_ids=np.concatenate([
            original.eviction_ids,
            new_data.eviction_ids[indices] + original.eviction_ids.max() + 1,
        ]),
        scenarios=np.concatenate([original.scenarios, new_data.scenarios[indices]]),
        feature_names=original.feature_names,
    )
