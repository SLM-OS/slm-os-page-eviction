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

    def test_non_standalone_emits_feature_schema_guard(self, tmp_path):
        """Non-standalone mode emits GENERATED_FEATURE_NAMES + uses
        &BlockFeatures in the signature.

        SLM-OS #952 Phase 2 relies on the const array being present so
        its compile-time schema guard (const_assert in
        runtime/src/mm/eviction/generated/mod.rs) can compare against
        the runtime's own FEATURE_NAMES. SLM-OS #957 relies on the
        signature using BlockFeatures so the matching `use` import
        isn't flagged unused under `-D warnings`. Both are silently
        regressable from this generator — pin them.

        Standalone mode is checked too (it must NOT emit the const,
        since standalone consumers don't have BlockFeatures in scope).
        """
        from src.export.xgb_to_rust import export_xgb_to_rust

        names = ["frequency_rank", "predicted_reuse_dist", "eviction_cost"]
        booster = self._train_tiny_model(names)

        slm_src = export_xgb_to_rust(
            booster, tmp_path / "slm.rs", feature_names=names, standalone=False,
        )
        standalone_src = export_xgb_to_rust(
            booster, tmp_path / "standalone.rs", feature_names=names, standalone=True,
        )

        # SLM-OS-side guard contract: const emitted, names in order,
        # length matches, signature uses the type alias.
        assert (
            f"pub const GENERATED_FEATURE_NAMES: [&str; {len(names)}] = ["
            in slm_src
        ), "GENERATED_FEATURE_NAMES const missing — SLM-OS #952 Phase 2 guard would fail"
        for name in names:
            assert f'"{name}",' in slm_src, f"feature name {name!r} not emitted in const"
        assert "pub fn xgb_predict(features: &BlockFeatures)" in slm_src, (
            "signature must use &BlockFeatures when import is present "
            "— SLM-OS #957 unused-import regression"
        )

        # Standalone mode: no const, inline-array signature.
        assert "GENERATED_FEATURE_NAMES" not in standalone_src, (
            "standalone mode must not emit GENERATED_FEATURE_NAMES — "
            "verification harnesses don't have BlockFeatures in scope"
        )
        assert (
            f"pub fn xgb_predict(features: &[f32; {len(names)}])"
            in standalone_src
        ), "standalone signature must keep the inline array form"

    def test_prune_slices_export_to_k_trees(self, tmp_path):
        """booster[0:K] exports exactly K trees — the mechanism behind
        `export_to_slmos.py --xgb-trees K` (SLM-OS #961). In gradient
        boosting the first K trees of an N-round model are a K-round
        model, so the slice is exact, not a retrain."""
        from src.export.xgb_to_rust import export_xgb_to_rust

        names = ["frequency_rank", "predicted_reuse_dist", "time_since_access"]
        booster = self._train_tiny_model(names)  # 5 boost rounds

        full = export_xgb_to_rust(booster, tmp_path / "full.rs", feature_names=names)
        pruned = export_xgb_to_rust(
            booster[0:2], tmp_path / "k2.rs", feature_names=names
        )

        assert "// Trees: 5," in full
        assert "// Trees: 2," in pruned
        # One `sum += {` block is emitted per tree.
        assert full.count("sum += {") == 5
        assert pruned.count("sum += {") == 2

    def test_prune_corpus_matches_sliced_model(self, tmp_path):
        """The verification corpus generated from a sliced booster matches
        that booster's own predictions — the lockstep guarantee that lets
        `--xgb-trees` keep the baked .rs and expected_evict.bin consistent
        (a 200-tree corpus against a 16-tree baked model would falsely fail
        `bench xgb-equiv-evict`)."""
        xgb = pytest.importorskip("xgboost")
        import numpy as np
        from src.export.xgb_to_smb import (
            EVICTION_FEATURE_COUNT,
            generate_evict_verification_corpus,
        )

        names = [f"f{i}" for i in range(EVICTION_FEATURE_COUNT)]
        sliced = self._train_tiny_model(names)[0:2]

        generate_evict_verification_corpus(
            sliced, tmp_path, feature_names=names, n_tests=64, seed=123
        )
        vecs = np.fromfile(
            tmp_path / "test_vectors_xgb_evict.bin", dtype=np.float32
        ).reshape(-1, EVICTION_FEATURE_COUNT)
        exp = np.fromfile(tmp_path / "expected_evict.bin", dtype=np.float32)
        got = sliced.predict(
            xgb.DMatrix(vecs, feature_names=names)
        ).astype(np.float32)

        assert exp.shape[0] == 64
        np.testing.assert_array_equal(exp, got)


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


# ---------------------------------------------------------------------------
# Pure-Python SEMB+XGB1 parser + walker. Mirrors SLM-OS's Rust
# `runtime/src/ml/xgb_tree.rs::predict_sigmoid` bit-for-bit; if either
# parser drifts the round-trip test in `TestXGBoostSmbExport` fails.
# ---------------------------------------------------------------------------

def _parse_smb_eviction_blob(blob: bytes) -> dict:
    """Return `{"trees": [[(feat, flags, left, right, thr, val), ...], ...],
    "roots": [int, ...]}` ready for `_walk_xgb1_sigmoid`."""
    import struct

    # SEMB outer (24 bytes).
    assert blob[0:4] == b"SEMB", "bad SEMB magic"
    (ver, kind, schema, _reserved_u16, payload_len) = struct.unpack(
        "<HHHHI", blob[4:16]
    )
    assert ver == 1 and kind == 1 and schema == 1, (ver, kind, schema)
    (checksum, _trailing) = struct.unpack("<II", blob[16:24])
    payload = blob[24:24 + payload_len]
    assert len(payload) == payload_len, "truncated payload"

    # Recompute FNV-1a and compare.
    h = 0x811C9DC5
    for b in payload:
        h ^= b
        h = (h * 0x01000193) & 0xFFFFFFFF
    assert h == checksum, f"checksum mismatch {h:x} != {checksum:x}"

    # XGB1 header (16 bytes).
    assert payload[0:4] == b"XGB1", "bad XGB1 magic"
    (pv, _ru16, n_trees, n_nodes, _ru32) = struct.unpack(
        "<HHHHI", payload[4:16]
    )
    assert pv == 1

    cursor = 16
    roots = list(struct.unpack(f"<{n_trees}H",
                               payload[cursor:cursor + 2 * n_trees]))
    cursor += 2 * n_trees

    nodes = []
    for _ in range(n_nodes):
        (feat, flags, left, right, thr, val) = struct.unpack(
            "<HHHHff", payload[cursor:cursor + 16]
        )
        nodes.append((feat, flags, left, right, thr, val))
        cursor += 16

    assert cursor == len(payload), "trailing bytes in payload"
    return {"roots": roots, "nodes": nodes}


def _walk_xgb1_sigmoid(parsed: dict, features) -> float:
    """Sum every tree's leaf contribution and sigmoid the total. Mirrors
    `XgbModel::predict_sigmoid` in SLM-OS."""
    import math

    total = 0.0
    nodes = parsed["nodes"]
    for root in parsed["roots"]:
        idx = root
        for _ in range(256):  # MAX_TREE_DEPTH safety cap
            feat, flags, left, right, thr, val = nodes[idx]
            if flags & 1:  # FLAG_LEAF
                total += val
                break
            f = features[feat]
            idx = left if f < thr else right
    return 1.0 / (1.0 + math.exp(-total))


class TestXGBoostSmbExport:
    """Regression tests for the XGB-to-SEMB blob exporter.

    The blob path complements the baked if-else path (`xgb_to_rust.py`)
    by letting researchers iterate on new eviction models without
    rebuilding the kernel. SLM-OS #932 motivates the verification
    surface; SLM-OS #920 motivates the base-margin fold.
    """

    def _train_tiny_model(self, n_features: int = 27, n_rounds: int = 5):
        xgb = pytest.importorskip("xgboost")
        import numpy as np

        rng = np.random.default_rng(0)
        n = 400
        X = rng.uniform(0, 1, size=(n, n_features)).astype(np.float32)
        # Mildly imbalanced binary target so XGBoost's auto-derived
        # base_score is non-trivial (logit != 0).
        y = (X[:, 0] + X[:, 1] > 1.3).astype(np.int8)
        dtrain = xgb.DMatrix(X, label=y)
        params = {
            "objective": "binary:logistic",
            "max_depth": 3,
            "eval_metric": "logloss",
            "verbosity": 0,
        }
        return xgb.train(params, dtrain, num_boost_round=n_rounds)

    def test_blob_has_semb_outer_header(self, tmp_path):
        from src.export.xgb_to_smb import (
            export_xgb_to_smb,
            SEMB_HEADER_LEN,
        )

        booster = self._train_tiny_model()
        blob = export_xgb_to_smb(booster, tmp_path / "evict.smb")
        assert blob[0:4] == b"SEMB"
        # Version=1, kind=1 (XGBoost), schema=1 — must match
        # `runtime/src/mm/eviction/blob.rs`.
        import struct
        ver, kind, schema = struct.unpack("<HHH", blob[4:10])
        assert (ver, kind, schema) == (1, 1, 1)
        assert len(blob) >= SEMB_HEADER_LEN

    def test_blob_xgb1_payload_round_trips(self, tmp_path):
        """Parse the blob with the test-side mirror parser; compare per-vector
        sigmoid output against booster.predict() within float tolerance."""
        import numpy as np

        from src.export.xgb_to_smb import export_xgb_to_smb

        booster = self._train_tiny_model()
        blob = export_xgb_to_smb(booster, tmp_path / "evict.smb")
        parsed = _parse_smb_eviction_blob(blob)

        # Generate 50 random feature vectors and compare predictions.
        rng = np.random.default_rng(42)
        X = rng.uniform(0, 1, size=(50, 27)).astype(np.float32)
        xgb = pytest.importorskip("xgboost")
        expected = booster.predict(xgb.DMatrix(X))

        # Per-vector check; max diff inside f32 sigmoid precision tolerance.
        max_diff = 0.0
        for i in range(50):
            got = _walk_xgb1_sigmoid(parsed, X[i].tolist())
            max_diff = max(max_diff, abs(got - float(expected[i])))
        # f32 leaf storage + Python-side f64 walk should agree to ~1e-5.
        assert max_diff < 1e-4, f"max abs diff {max_diff}"

    def test_synthetic_base_margin_tree_is_prepended(self, tmp_path):
        """Tree count must be `real_trees + 1`; the new first tree is a
        single-leaf node whose value equals logit(base_score). Catches a
        future maintainer removing or moving the base-score fold."""
        import math

        from src.export.xgb_to_smb import export_xgb_to_smb

        booster = self._train_tiny_model(n_rounds=4)
        blob = export_xgb_to_smb(booster, tmp_path / "evict.smb")
        parsed = _parse_smb_eviction_blob(blob)

        # Booster has 4 real trees → blob has 5 (4 + 1 synthetic).
        assert len(parsed["roots"]) == 5
        # Synthetic tree at root[0] is a single leaf with value =
        # logit(base_score). Read base_score back from the model.
        import json
        cfg = json.loads(booster.save_config())
        bs_raw = cfg["learner"]["learner_model_param"].get("base_score", "0.5")
        if isinstance(bs_raw, str) and bs_raw.startswith("["):
            base_score = float(bs_raw.strip("[]").split(",")[0])
        else:
            base_score = float(bs_raw)
        expected_margin = math.log(base_score / (1.0 - base_score))

        synth_root = parsed["roots"][0]
        synth_node = parsed["nodes"][synth_root]
        feat, flags, _l, _r, _thr, val = synth_node
        assert flags & 1, "first tree's root should be a leaf"
        assert abs(val - expected_margin) < 1e-5, (val, expected_margin)

    def test_blob_rejects_oversized_feature_idx(self, tmp_path):
        """A tree that splits on feature 99 should fail at export time, not
        produce a runtime parser rejection (which is harder to debug)."""
        from src.export.xgb_to_smb import _flatten_tree

        node = {
            "split": "f99",
            "split_condition": 0.5,
            "yes": 1, "no": 2,
            "nodeid": 0,
            "children": [
                {"nodeid": 1, "leaf": 0.1},
                {"nodeid": 2, "leaf": -0.1},
            ],
        }
        with pytest.raises(ValueError, match="feature_idx 99"):
            _flatten_tree(node, feature_index={})

    def test_generate_evict_corpus_shapes(self, tmp_path):
        """Corpus byte sizes must match `N × FEATURE × f32` and `N × f32` so
        the on-device verb can mmap-style read them by file length."""
        from src.export.xgb_to_smb import (
            generate_evict_verification_corpus,
            EVICTION_FEATURE_COUNT,
        )

        booster = self._train_tiny_model()
        generate_evict_verification_corpus(
            booster, tmp_path, n_tests=37,
        )
        vec = (tmp_path / "test_vectors_xgb_evict.bin").read_bytes()
        exp = (tmp_path / "expected_evict.bin").read_bytes()
        assert len(vec) == 37 * EVICTION_FEATURE_COUNT * 4
        assert len(exp) == 37 * 4
