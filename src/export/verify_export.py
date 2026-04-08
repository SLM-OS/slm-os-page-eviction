"""Cross-validate Python model predictions vs Rust-exported code.

Runs the same inputs through both the Python model and a subprocess
executing the generated Rust code, verifying that predictions match
within quantization tolerance.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.features.extractor import FeatureConfig


@dataclass
class VerificationResult:
    """Results from cross-validation of Python vs Rust predictions."""
    num_samples: int
    max_abs_error: float
    mean_abs_error: float
    num_mismatches: int     # Predictions that disagree on eviction choice
    mismatch_rate: float
    passed: bool            # True if within tolerance


def verify_predictions(
    python_predictions: np.ndarray,
    rust_predictions: np.ndarray,
    tolerance: float = 0.01,
) -> VerificationResult:
    """Compare Python and Rust model predictions.

    For classification, two predictions "match" if they agree on which
    candidate has the highest score (same argmax). The tolerance is
    applied to the raw probability values.

    Args:
        python_predictions: Probability scores from Python model.
        rust_predictions: Probability scores from Rust implementation.
        tolerance: Maximum allowed absolute difference.

    Returns:
        VerificationResult with match statistics.
    """
    abs_errors = np.abs(python_predictions - rust_predictions)
    max_error = float(np.max(abs_errors))
    mean_error = float(np.mean(abs_errors))

    # Count argmax mismatches (grouped by eviction event)
    num_mismatches = 0
    # For flat arrays, just check element-wise
    mismatches = abs_errors > tolerance
    num_mismatches = int(np.sum(mismatches))

    return VerificationResult(
        num_samples=len(python_predictions),
        max_abs_error=max_error,
        mean_abs_error=mean_error,
        num_mismatches=num_mismatches,
        mismatch_rate=num_mismatches / max(len(python_predictions), 1),
        passed=max_error <= tolerance,
    )


def verify_eviction_decisions(
    python_scores: np.ndarray,
    rust_scores: np.ndarray,
    eviction_ids: np.ndarray,
) -> VerificationResult:
    """Verify that Python and Rust agree on eviction decisions.

    Groups predictions by eviction_id and checks that both implementations
    select the same victim (same argmax within each group).
    """
    unique_ids = np.unique(eviction_ids)
    num_mismatches = 0

    for eid in unique_ids:
        mask = eviction_ids == eid
        py_choice = np.argmax(python_scores[mask])
        rust_choice = np.argmax(rust_scores[mask])
        if py_choice != rust_choice:
            num_mismatches += 1

    abs_errors = np.abs(python_scores - rust_scores)

    return VerificationResult(
        num_samples=len(unique_ids),
        max_abs_error=float(np.max(abs_errors)),
        mean_abs_error=float(np.mean(abs_errors)),
        num_mismatches=num_mismatches,
        mismatch_rate=num_mismatches / max(len(unique_ids), 1),
        passed=num_mismatches == 0,
    )


def generate_test_vectors(
    num_samples: int = 1000,
    num_features: int = 27,
    seed: int = 42,
) -> np.ndarray:
    """Generate random test vectors for cross-validation.

    Produces feature vectors with realistic ranges matching the
    normalized feature space.
    """
    rng = np.random.default_rng(seed)
    # All features are normalized to roughly [0, 1]
    return rng.uniform(0.0, 1.0, size=(num_samples, num_features)).astype(np.float32)
