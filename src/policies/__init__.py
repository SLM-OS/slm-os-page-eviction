from .base import EvictionPolicy
from .lru import LRUPolicy
from .lfu import LFUPolicy
from .arc import ARCPolicy
from .slm_heuristic import SLMHeuristicPolicy
from .belady import BeladyOracle

# ML-dependent policies: import lazily to avoid requiring torch/xgboost
# for simulator-only usage.
__all__ = [
    "EvictionPolicy",
    "LRUPolicy", "LFUPolicy", "ARCPolicy", "SLMHeuristicPolicy",
    "BeladyOracle",
]


def __getattr__(name: str):
    if name == "MLPPolicy":
        from .mlp_policy import MLPPolicy
        return MLPPolicy
    if name == "XGBPolicy":
        from .xgb_policy import XGBPolicy
        return XGBPolicy
    if name == "CACHEUSSelector":
        from .cacheus import CACHEUSSelector
        return CACHEUSSelector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
