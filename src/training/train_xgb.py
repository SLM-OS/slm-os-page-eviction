"""XGBoost training pipeline (Section 7.1 and 8.1 of the plan).

Trains an XGBoost classifier to predict the Belady-optimal eviction
target. The model outputs P(optimal eviction target) for each candidate;
at inference time, the candidate with the highest probability is evicted.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import xgboost as xgb

from .dataset import EvictionDataset


@dataclass
class XGBTrainConfig:
    """XGBoost hyperparameters (Section 7.1 table)."""
    objective: str = "binary:logistic"
    num_rounds: int = 200
    max_depth: int = 6
    learning_rate: float = 0.1
    min_child_weight: int = 10
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    scale_pos_weight: float = 30.0  # ~1 optimal per ~30 candidates
    eval_metric: list[str] | None = None
    early_stopping_rounds: int = 20
    seed: int = 42

    def __post_init__(self):
        if self.eval_metric is None:
            self.eval_metric = ["auc", "logloss"]

    def to_xgb_params(self) -> dict[str, Any]:
        """Convert to xgboost parameter dict."""
        return {
            "objective": self.objective,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "min_child_weight": self.min_child_weight,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "scale_pos_weight": self.scale_pos_weight,
            "eval_metric": self.eval_metric,
            "seed": self.seed,
            "verbosity": 1,
        }


@dataclass
class XGBTrainResult:
    """Results from XGBoost training."""
    model: xgb.Booster
    train_auc: float
    val_auc: float
    best_iteration: int
    feature_importance: dict[str, float]  # feature_name -> gain
    evals_result: dict[str, Any]


def train_xgboost(
    train_data: EvictionDataset,
    val_data: EvictionDataset,
    config: XGBTrainConfig | None = None,
) -> XGBTrainResult:
    """Train an XGBoost model for eviction prediction.

    Args:
        train_data: Training dataset with Belady-optimal labels.
        val_data: Validation dataset for early stopping.
        config: Training hyperparameters.

    Returns:
        XGBTrainResult with trained model and metrics.
    """
    cfg = config or XGBTrainConfig()

    # Compute dynamic scale_pos_weight from actual class balance
    pos_count = np.sum(train_data.is_optimal == 1)
    neg_count = np.sum(train_data.is_optimal == 0)
    if pos_count > 0:
        cfg.scale_pos_weight = neg_count / pos_count

    dtrain = xgb.DMatrix(
        train_data.features,
        label=train_data.is_optimal,
        feature_names=train_data.feature_names,
    )
    dval = xgb.DMatrix(
        val_data.features,
        label=val_data.is_optimal,
        feature_names=val_data.feature_names,
    )

    evals_result: dict[str, Any] = {}
    model = xgb.train(
        cfg.to_xgb_params(),
        dtrain,
        num_boost_round=cfg.num_rounds,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=cfg.early_stopping_rounds,
        evals_result=evals_result,
        verbose_eval=10,
    )

    # Extract feature importance
    importance = model.get_score(importance_type="gain")

    # Compute final AUC
    train_auc = float(evals_result["train"]["auc"][-1])
    val_auc = float(evals_result["val"]["auc"][-1])

    return XGBTrainResult(
        model=model,
        train_auc=train_auc,
        val_auc=val_auc,
        best_iteration=model.best_iteration,
        feature_importance=importance,
        evals_result=evals_result,
    )


def cross_validate_xgb(
    dataset: EvictionDataset,
    config: XGBTrainConfig | None = None,
    n_folds: int = 5,
) -> list[float]:
    """K-fold cross-validation by scenario (not by row).

    Folds are constructed by scenario to measure generalization
    across different workload types, not just different eviction events.

    Returns:
        List of validation AUC scores, one per fold.
    """
    cfg = config or XGBTrainConfig()
    scenarios = np.unique(dataset.scenarios)
    rng = np.random.default_rng(cfg.seed)
    rng.shuffle(scenarios)

    # Split scenarios into folds
    fold_scenarios = np.array_split(scenarios, n_folds)
    aucs = []

    for fold_idx in range(n_folds):
        val_scenarios = set(fold_scenarios[fold_idx])
        val_mask = np.isin(dataset.scenarios, list(val_scenarios))
        train_mask = ~val_mask

        if not np.any(train_mask) or not np.any(val_mask):
            continue

        train_split = EvictionDataset(
            features=dataset.features[train_mask],
            is_optimal=dataset.is_optimal[train_mask],
            reuse_distance=dataset.reuse_distance[train_mask],
            eviction_ids=dataset.eviction_ids[train_mask],
            scenarios=dataset.scenarios[train_mask],
            feature_names=dataset.feature_names,
        )
        val_split = EvictionDataset(
            features=dataset.features[val_mask],
            is_optimal=dataset.is_optimal[val_mask],
            reuse_distance=dataset.reuse_distance[val_mask],
            eviction_ids=dataset.eviction_ids[val_mask],
            scenarios=dataset.scenarios[val_mask],
            feature_names=dataset.feature_names,
        )

        result = train_xgboost(train_split, val_split, cfg)
        aucs.append(result.val_auc)

    return aucs


def grid_search_xgb(
    train_data: EvictionDataset,
    val_data: EvictionDataset,
    param_grid: dict[str, list] | None = None,
) -> list[dict]:
    """Grid search over XGBoost hyperparameters.

    Args:
        train_data: Training dataset.
        val_data: Validation dataset.
        param_grid: Dict of param_name -> list of values to try.
            Defaults to the grid from configs/xgb_params.yaml.

    Returns:
        List of dicts with params and val_auc, sorted by val_auc descending.
    """
    if param_grid is None:
        param_grid = {
            "max_depth": [4, 5, 6, 7],
            "learning_rate": [0.05, 0.1, 0.2],
            "min_child_weight": [5, 10, 20],
        }

    # Generate all combinations
    import itertools
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combos = list(itertools.product(*values))

    results = []
    for i, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        cfg = XGBTrainConfig(**params)
        result = train_xgboost(train_data, val_data, cfg)
        entry = {**params, "val_auc": result.val_auc, "best_iteration": result.best_iteration}
        results.append(entry)
        print(f"  [{i+1}/{len(combos)}] {params} -> val_auc={result.val_auc:.6f} (iter {result.best_iteration})")

    results.sort(key=lambda x: x["val_auc"], reverse=True)
    return results


def save_model(model: xgb.Booster, path: str | Path) -> None:
    """Save trained XGBoost model."""
    model.save_model(str(path))


def load_model(path: str | Path) -> xgb.Booster:
    """Load a trained XGBoost model."""
    model = xgb.Booster()
    model.load_model(str(path))
    return model
