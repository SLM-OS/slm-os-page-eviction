"""XGBoost eviction policy wrapper (Section 7.1 of the plan).

Wraps the trained XGBoost model as an EvictionPolicy for use in
the simulator and as a CACHEUS expert.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import xgboost as xgb

from .base import EvictionPolicy
from src.features.extractor import FeatureConfig, FeatureExtractor
from src.features.normalizer import FeatureNormalizer

if TYPE_CHECKING:
    from src.simulator.block import BlockMeta
    from src.simulator.core import GlobalState, Pool


class XGBPolicy(EvictionPolicy):
    """Eviction policy using the trained XGBoost model.

    Extracts features for each candidate, runs them through XGBoost,
    and evicts the candidate with the highest predicted probability
    of being the optimal eviction target.
    """

    def __init__(
        self,
        model: xgb.Booster,
        feature_config: FeatureConfig | None = None,
    ):
        self._model = model
        self._config = feature_config or FeatureConfig()
        self._extractor = FeatureExtractor(self._config)
        self._normalizer = FeatureNormalizer(self._config)

    def select_victim(
        self,
        candidates: list[BlockMeta],
        pool: Pool,
        global_state: GlobalState,
    ) -> int:
        """Select victim by highest XGBoost classification probability."""
        scores = self.score(candidates, pool, global_state)
        return int(np.argmax(scores))

    def score(
        self,
        candidates: list[BlockMeta],
        pool: Pool,
        global_state: GlobalState,
    ) -> list[float]:
        """Score each candidate using the XGBoost model."""
        if not candidates:
            return []

        features = self._extractor.extract_candidate_features(
            candidates, pool, global_state
        )
        features = self._normalizer.normalize(
            features, pool_size=pool.num_blocks
        )

        dmatrix = xgb.DMatrix(
            features,
            feature_names=self._config.feature_names,
        )
        probs = self._model.predict(dmatrix)
        return probs.tolist()

    def name(self) -> str:
        return "XGBoost"
