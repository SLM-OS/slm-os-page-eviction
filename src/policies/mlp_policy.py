"""MLP eviction policy wrapper (Section 7.2 of the plan).

Wraps the trained PageReplacementMLP model as an EvictionPolicy for
use in the simulator and as a CACHEUS expert.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from .base import EvictionPolicy
from src.features.extractor import FeatureConfig, FeatureExtractor
from src.features.normalizer import FeatureNormalizer

if TYPE_CHECKING:
    from src.simulator.block import BlockMeta
    from src.simulator.core import GlobalState, Pool
    from src.training.train_mlp import PageReplacementMLP


class MLPPolicy(EvictionPolicy):
    """Eviction policy using the trained MLP model.

    Extracts features for each candidate, runs them through the MLP,
    and evicts the candidate with the highest predicted probability
    of being the optimal eviction target.
    """

    def __init__(
        self,
        model: PageReplacementMLP,
        feature_config: FeatureConfig | None = None,
    ):
        self._model = model
        self._model.eval()
        self._extractor = FeatureExtractor(feature_config)
        self._normalizer = FeatureNormalizer(feature_config)
        self._config = feature_config or FeatureConfig()

    @property
    def feature_config(self) -> FeatureConfig:
        """The feature configuration used by this policy."""
        return self._config

    def update_model(self, model: PageReplacementMLP) -> None:
        """Replace the underlying model (used during DAgger fine-tuning)."""
        self._model = model
        self._model.eval()

    def select_victim(
        self,
        candidates: list[BlockMeta],
        pool: Pool,
        global_state: GlobalState,
    ) -> int:
        """Select victim by highest MLP classification probability."""
        scores = self.score(candidates, pool, global_state)
        return int(np.argmax(scores))

    def score(
        self,
        candidates: list[BlockMeta],
        pool: Pool,
        global_state: GlobalState,
    ) -> list[float]:
        """Score each candidate using the MLP model."""
        if not candidates:
            return []

        features = self._extractor.extract_candidate_features(
            candidates, pool, global_state
        )
        features = self._normalizer.normalize(
            features, pool_size=pool.num_blocks
        )

        with torch.no_grad():
            tensor = torch.from_numpy(features).float()
            probs = self._model.predict_scores(tensor)
            return probs.numpy().tolist()

    def name(self) -> str:
        return "MLP"
