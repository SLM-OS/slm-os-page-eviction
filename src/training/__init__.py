from .dataset import EvictionDataset, load_dataset, train_val_test_split
from .evaluate import PolicyEvaluator

# ML-dependent modules: import lazily to avoid requiring torch/xgboost
# for dataset and evaluation-only usage.
__all__ = [
    "EvictionDataset", "load_dataset", "train_val_test_split",
    "PolicyEvaluator",
]


def __getattr__(name: str):
    if name == "train_xgboost":
        from .train_xgb import train_xgboost
        return train_xgboost
    if name in ("train_mlp", "PageReplacementMLP"):
        from . import train_mlp as _mod
        return getattr(_mod, name)
    if name == "dagger_finetune":
        from .dagger import dagger_finetune
        return dagger_finetune
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
