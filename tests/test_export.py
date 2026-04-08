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
