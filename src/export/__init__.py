from .verify_export import verify_predictions

# ML-dependent exports: import lazily to avoid requiring torch/xgboost
# for verification-only usage.
__all__ = ["verify_predictions"]


def __getattr__(name: str):
    if name == "export_xgb_to_rust":
        from .xgb_to_rust import export_xgb_to_rust
        return export_xgb_to_rust
    if name == "export_mlp_to_rust":
        from .mlp_to_rust import export_mlp_to_rust
        return export_mlp_to_rust
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
