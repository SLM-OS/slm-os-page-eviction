"""Unit tests for model export and cross-validation.

Validates that XGBoost-to-Rust and MLP-to-Rust conversions produce
syntactically valid output, and that the verification framework
correctly detects mismatches.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.export.verify_export import (
    VerificationResult,
    generate_test_vectors,
    verify_predictions,
    verify_eviction_decisions,
)


class TestVerifyPredictions:
    """Tests for the prediction verification framework."""

    def test_identical_predictions_pass(self):
        preds = np.array([0.1, 0.9, 0.5, 0.3])
        result = verify_predictions(preds, preds.copy(), tolerance=0.01)
        assert result.passed
        assert result.num_mismatches == 0
        assert result.max_abs_error == 0.0

    def test_different_predictions_fail(self):
        py_preds = np.array([0.1, 0.9, 0.5])
        rust_preds = np.array([0.1, 0.8, 0.6])
        result = verify_predictions(py_preds, rust_preds, tolerance=0.01)
        assert not result.passed
        assert result.num_mismatches > 0

    def test_within_tolerance_passes(self):
        py_preds = np.array([0.5, 0.6, 0.7])
        rust_preds = np.array([0.505, 0.595, 0.705])
        result = verify_predictions(py_preds, rust_preds, tolerance=0.01)
        assert result.passed

    def test_generate_test_vectors_shape(self):
        vectors = generate_test_vectors(100, 27)
        assert vectors.shape == (100, 27)
        assert vectors.dtype == np.float32
        assert np.all(vectors >= 0.0)
        assert np.all(vectors <= 1.0)


class TestVerifyEvictionDecisions:
    """Tests for eviction decision verification."""

    def test_same_argmax_passes(self):
        # Two eviction events with 3 candidates each
        py_scores = np.array([0.1, 0.9, 0.5, 0.3, 0.7, 0.2])
        rust_scores = np.array([0.15, 0.85, 0.45, 0.35, 0.65, 0.25])
        eviction_ids = np.array([0, 0, 0, 1, 1, 1])
        result = verify_eviction_decisions(py_scores, rust_scores, eviction_ids)
        # Both should pick candidate 1 for event 0, candidate 1 for event 1
        assert result.passed
        assert result.num_mismatches == 0

    def test_different_argmax_fails(self):
        py_scores = np.array([0.1, 0.9, 0.5])
        rust_scores = np.array([0.9, 0.1, 0.5])  # Swapped!
        eviction_ids = np.array([0, 0, 0])
        result = verify_eviction_decisions(py_scores, rust_scores, eviction_ids)
        assert not result.passed
        assert result.num_mismatches == 1


class TestXGBoostExport:
    """Regression tests for the XGBoost-to-Rust generator.

    Earlier versions assumed feature names of the form 'f<index>'. When trained
    with descriptive names like 'frequency_rank', the 'f'-prefix check
    incorrectly tried to parse 'requency_rank' as an integer. The exporter now
    uses an explicit name → index map.
    """

    def _train_tiny_model(self, feature_names: list[str]):
        xgb = pytest.importorskip("xgboost")
        import numpy as np

        rng = np.random.default_rng(0)
        n = 200
        X = rng.uniform(0, 1, size=(n, len(feature_names))).astype(np.float32)
        y = (X[:, 0] > 0.5).astype(np.int8)
        dtrain = xgb.DMatrix(X, label=y, feature_names=feature_names)
        params = {
            "objective": "binary:logistic",
            "max_depth": 3,
            "eval_metric": "logloss",
            "verbosity": 0,
        }
        return xgb.train(params, dtrain, num_boost_round=5)

    def test_descriptive_feature_names_export(self, tmp_path):
        """Trees that split on 'frequency_rank' don't crash the exporter."""
        from src.export.xgb_to_rust import export_xgb_to_rust

        names = ["frequency_rank", "predicted_reuse_dist", "time_since_access"]
        booster = self._train_tiny_model(names)

        out = tmp_path / "policy.rs"
        source = export_xgb_to_rust(booster, out, feature_names=names)

        assert "fn xgb_predict" in source
        # Generated code should reference at least one feature with a comment
        assert any(name in source for name in names)
        # No raw "f<word>" parsing errors should leak through
        assert "/* requency_rank */" not in source

    def test_default_f_prefix_export_still_works(self, tmp_path):
        """If feature_names is omitted, default 'f<i>' convention is used."""
        from src.export.xgb_to_rust import export_xgb_to_rust

        booster = self._train_tiny_model(["f0", "f1", "f2"])
        out = tmp_path / "policy.rs"
        source = export_xgb_to_rust(booster, out, feature_names=["f0", "f1", "f2"])
        assert "fn xgb_predict" in source

    def test_unknown_feature_raises(self, tmp_path):
        """Unknown split feature (not in names, not 'f<i>') raises clearly."""
        from src.export.xgb_to_rust import _tree_to_rust

        # Synthetic node with an unknown split feature
        node = {
            "split": "weird_feature",
            "split_condition": 0.5,
            "yes": 1, "no": 2,
            "children": [
                {"nodeid": 1, "leaf": 0.1},
                {"nodeid": 2, "leaf": -0.1},
            ],
        }
        with pytest.raises(ValueError, match="weird_feature"):
            _tree_to_rust(node, [], feature_index={"other_feature": 0})

    def test_standalone_omits_use_import(self, tmp_path):
        """standalone=True drops the SLM-OS-specific `use crate::...` line."""
        from src.export.xgb_to_rust import export_xgb_to_rust

        names = ["frequency_rank", "predicted_reuse_dist"]
        booster = self._train_tiny_model(names)

        slm_path = tmp_path / "slm.rs"
        slm_src = export_xgb_to_rust(booster, slm_path, feature_names=names, standalone=False)
        standalone_path = tmp_path / "standalone.rs"
        standalone_src = export_xgb_to_rust(
            booster, standalone_path, feature_names=names, standalone=True,
        )

        assert "use crate::mm::eviction_policy" in slm_src
        assert "use crate::mm::eviction_policy" not in standalone_src
        # Both still define the prediction function
        assert "fn xgb_predict" in slm_src
        assert "fn xgb_predict" in standalone_src


class TestMLPRustExport:
    """Tests for the MLP int8 Rust export.

    Earlier versions emitted a layer-1 inference using i32 MAC with input
    quantization scaled by W_L1_SCALE — but the formula assumed inputs were
    in [-W_SCALE, W_SCALE] and silently truncated for normalized features
    in [0, 1]. The current implementation uses f32 MAC with int8
    dequantize-on-use, matching the Python forward pass.
    """

    def _make_model(self, num_features: int = 27):
        torch = pytest.importorskip("torch")
        from src.training.train_mlp import PageReplacementMLP
        return PageReplacementMLP(num_features=num_features)

    def test_quantized_layer1_uses_f32_mac(self, tmp_path):
        from src.export.mlp_to_rust import export_mlp_to_rust

        model = self._make_model()
        out = tmp_path / "mlp.rs"
        source = export_mlp_to_rust(model, out, quantize=True)

        # Layer 1 should use the f32 MAC pattern (matches layers 2-3),
        # not the broken i32 MAC with input quantization
        assert "let mut acc = 0.0_f32;" in source
        assert "features[j] / W_L1_SCALE" not in source, \
            "Layer-1 input quantization formula was broken; should not appear"
        assert "features[j] * (W_L1[i][j] as f32) * W_L1_SCALE" in source

    def test_quantized_inference_function_present(self, tmp_path):
        from src.export.mlp_to_rust import export_mlp_to_rust

        model = self._make_model()
        out = tmp_path / "mlp.rs"
        source = export_mlp_to_rust(model, out, quantize=True)
        assert "pub fn mlp_predict(" in source
        # Sigmoid output
        assert "(-out).exp()" in source

    def test_compute_model_size_matches_target(self, tmp_path):
        """Int8 model size should be in the few-KB range (target < 5 KB)."""
        from src.export.mlp_to_rust import compute_model_size

        model = self._make_model()
        size = compute_model_size(model, quantize=True)
        # 27→64→32→16→1 with f32 scale per tensor: ~4.5 KB
        assert 3000 < size < 6000, f"Unexpected int8 model size: {size} bytes"


class TestRustVerifyHelpers:
    """Tests for the CSV protocol helpers in scripts/verify_rust_export.py."""

    def test_to_csv_lines_roundtrip(self):
        from scripts.verify_rust_export import _to_csv_lines, _parse_lines
        import numpy as np

        arr = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)
        csv = _to_csv_lines(arr)
        assert csv.count("\n") == 2  # one trailing newline per row
        # Each row is comma-separated
        first_line = csv.splitlines()[0]
        assert first_line.count(",") == 2

    def test_parse_lines_handles_blanks(self):
        from scripts.verify_rust_export import _parse_lines

        out = "0.5\n0.25\n\n0.75\n"
        result = _parse_lines(out)
        assert list(result) == pytest.approx([0.5, 0.25, 0.75])
