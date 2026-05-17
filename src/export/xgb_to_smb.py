"""Convert trained XGBoost model to an SEMB+XGB1 blob for SLM-OS.

This is the blob-loader counterpart to `xgb_to_rust.py`. The if-else
Rust file ships baked into the kernel binary; the SEMB blob ships
separately on the SD card and is staged at runtime via
`rust_eviction_blob_stage`. SLM-OS's `XGBoostPolicy::score_row`
prefers the blob when one is present and falls back to the baked
function otherwise — so the two paths coexist and the blob path
enables experimentation without rebuilding the kernel.

Wire format: see `runtime/src/mm/eviction/blob.rs` (outer SEMB
header) and `runtime/src/ml/xgb_tree.rs` (XGB1 single-classifier
payload, 16-byte node, u16 children).

Base-score handling: the on-device walker
(`XgbModel::predict_sigmoid`) sums every tree's leaf contribution and
applies sigmoid. It has no separate base-margin term. XGBoost's
`booster.predict()` does `sigmoid(base_margin + sum_of_trees)` where
`base_margin = logit(base_score)`. To make the two agree, we prepend
one synthetic single-leaf tree with `value = logit(base_score)` at
tree index 0 so it folds into the same `sum`. See SLM-OS #920 for
the analogous scheduler-side fold + rationale.
"""

from __future__ import annotations

import json
import math
import struct
from pathlib import Path
from typing import Any

import xgboost as xgb


# ---------------------------------------------------------------------------
# Wire-format constants — must mirror SLM-OS exactly. Any drift here
# silently produces blobs the on-device parser rejects (best case) or
# silently mis-parses (worst case).
# ---------------------------------------------------------------------------

SEMB_MAGIC = b"SEMB"
SEMB_VERSION_V1 = 1
SEMB_KIND_XGBOOST = 1            # `BlobKind::XGBoost = 1`
SEMB_FEATURE_SCHEMA_V1 = 1
SEMB_HEADER_LEN = 24

XGB1_MAGIC = b"XGB1"
XGB1_PAYLOAD_VERSION_V1 = 1
XGB1_HEADER_LEN = 16
XGB1_NODE_LEN = 16
XGB1_FLAG_LEAF = 1

# Runtime caps from `runtime/src/ml/xgb_tree.rs`. Surface as ValueError
# at export time rather than letting the kernel reject the blob.
XGB1_MAX_NODES = 65_535          # u16 wire field
XGB1_MAX_TREES = 4_096           # MAX_TREES_SINGLE

# Eviction's `RuntimeXGBoostModel` parses with `max_feature_idx = 27`.
# A model trained with a different feature schema would deploy but
# silently reference the wrong slots; catch at export time.
EVICTION_FEATURE_COUNT = 27


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fnv1a_32(data: bytes) -> int:
    """32-bit FNV-1a. Must match `runtime/src/mm/eviction/blob.rs::checksum32`."""
    h = 0x811C9DC5
    for b in data:
        h ^= b
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


def _logit(p: float) -> float:
    """Numerically safe logit, clamping away from {0, 1} so log() stays
    finite even when a poorly-balanced training set yields a
    near-degenerate base. Mirrors `_xgb_logit` in slm-os-scheduler-ai."""
    eps = 1e-12
    p = max(min(p, 1.0 - eps), eps)
    return math.log(p / (1.0 - p))


def _parse_base_score(bs_raw: Any) -> float:
    """Parse XGBoost's `base_score` config field into a float.

    For binary classifiers XGBoost stores either a string-encoded
    probability (`"4.16e-1"`) or a 1-element bracketed list
    (`"[4.16e-1]"`); occasionally a native Python numeric leaks
    through. Try real JSON first so a future XGBoost release that
    switches encoding or adds whitespace doesn't silently break the
    fold. Mirrors `_xgb_parse_base_score` in slm-os-scheduler-ai.
    """
    if isinstance(bs_raw, list):
        bs_vec = [float(x) for x in bs_raw]
    elif isinstance(bs_raw, (int, float)):
        bs_vec = [float(bs_raw)]
    else:
        try:
            parsed = json.loads(bs_raw)
        except (json.JSONDecodeError, TypeError):
            return float(bs_raw)
        if isinstance(parsed, list):
            bs_vec = [float(x) for x in parsed]
        elif isinstance(parsed, (int, float)):
            bs_vec = [float(parsed)]
        else:
            return float(bs_raw)
    if len(bs_vec) != 1:
        raise ValueError(
            f"binary base_score must be a single value, got {bs_vec}"
        )
    return bs_vec[0]


def _flatten_tree(
    root: dict[str, Any],
    feature_index: dict[str, int],
) -> list[tuple[int, int, int, int, float, float]]:
    """Flatten one XGBoost tree-dict into a list of nodes in pre-order.

    Each node tuple is `(feature_idx, flags, left, right, threshold,
    value)`. Child indices are local to the returned list — the caller
    re-bases them when assembling the flat XGB1 node array.
    """
    nodes: list[Any] = []

    def walk(node: dict[str, Any]) -> int:
        my_idx = len(nodes)
        if "leaf" in node:
            nodes.append((0, XGB1_FLAG_LEAF, 0, 0, 0.0, float(node["leaf"])))
            return my_idx

        # Interior node — reserve the slot, recurse, fill in.
        nodes.append(None)
        split_feature = node.get("split", "f0")
        if split_feature in feature_index:
            feat_idx = feature_index[split_feature]
        elif split_feature.startswith("f") and split_feature[1:].isdigit():
            feat_idx = int(split_feature[1:])
        else:
            raise ValueError(
                f"unknown split feature {split_feature!r} — not in "
                f"feature_names and not in default 'f<i>' form"
            )
        if feat_idx >= EVICTION_FEATURE_COUNT:
            raise ValueError(
                f"feature_idx {feat_idx} ({split_feature!r}) exceeds "
                f"runtime feature count {EVICTION_FEATURE_COUNT}"
            )
        threshold = float(node.get("split_condition", 0.0))
        yes_id = node["yes"]
        no_id = node["no"]
        children = node.get("children", [])
        yes_node = next((c for c in children if c["nodeid"] == yes_id), None)
        no_node = next((c for c in children if c["nodeid"] == no_id), None)
        if yes_node is None or no_node is None:
            raise ValueError(
                f"interior node {node.get('nodeid')} missing child "
                f"(yes={yes_id}, no={no_id})"
            )
        left = walk(yes_node)
        right = walk(no_node)
        nodes[my_idx] = (feat_idx, 0, left, right, threshold, 0.0)
        return my_idx

    walk(root)
    return nodes


def _synthetic_base_margin_tree(
    margin: float,
) -> list[tuple[int, int, int, int, float, float]]:
    """One-node tree carrying the base-margin offset as a leaf value.
    Prepended at tree index 0 so the runtime's tree-sum picks it up."""
    return [(0, XGB1_FLAG_LEAF, 0, 0, 0.0, float(margin))]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def export_xgb_to_smb(
    model: xgb.Booster,
    output_path: str | Path,
    feature_names: list[str] | None = None,
) -> bytes:
    """Serialize a trained binary XGBoost classifier into an SEMB+XGB1
    blob consumable by SLM-OS's eviction `RuntimeXGBoostModel`.

    Args:
        model: Trained `xgb.Booster` (binary:logistic).
        output_path: Destination `.smb` file path.
        feature_names: Feature names matching SLM-OS's `BlockFeatures`
            layout. Defaults to `[f0..fN-1]` if omitted; only safe when
            the booster was trained without named features.

    Returns:
        The blob bytes. Also written to `output_path`.
    """
    feature_names = feature_names or [
        f"f{i}" for i in range(EVICTION_FEATURE_COUNT)
    ]
    if len(feature_names) != EVICTION_FEATURE_COUNT:
        raise ValueError(
            f"feature_names length {len(feature_names)} != runtime "
            f"feature count {EVICTION_FEATURE_COUNT}"
        )
    feature_index = {n: i for i, n in enumerate(feature_names)}

    # Fold base_score into a synthetic leaf tree at index 0.
    cfg = json.loads(model.save_config())
    bs_raw = cfg["learner"]["learner_model_param"].get("base_score", "0.5")
    base_score = _parse_base_score(bs_raw)
    base_margin = _logit(base_score)

    # Flatten real trees.
    tree_dump = model.get_dump(dump_format="json")
    real_trees = [
        _flatten_tree(json.loads(t), feature_index) for t in tree_dump
    ]

    all_trees = [_synthetic_base_margin_tree(base_margin)] + real_trees

    # Assemble flat node array + per-tree root offsets, re-basing child
    # indices from each tree-local frame into the flat array.
    flat_nodes: list[tuple[int, int, int, int, float, float]] = []
    tree_roots: list[int] = []
    for tree in all_trees:
        base_off = len(flat_nodes)
        tree_roots.append(base_off)
        for (feat_idx, flags, left, right, threshold, value) in tree:
            if (flags & XGB1_FLAG_LEAF) == 0:
                left += base_off
                right += base_off
            flat_nodes.append(
                (feat_idx, flags, left, right, threshold, value)
            )

    n_trees = len(all_trees)
    n_nodes = len(flat_nodes)
    if n_trees > XGB1_MAX_TREES:
        raise ValueError(
            f"{n_trees} trees exceeds XGB1 runtime cap {XGB1_MAX_TREES} "
            f"(bump MAX_TREES_SINGLE in xgb_tree.rs)"
        )
    if n_nodes > XGB1_MAX_NODES:
        raise ValueError(
            f"{n_nodes} nodes exceeds XGB1 u16 cap {XGB1_MAX_NODES} — "
            f"the eviction wire format can't address this many nodes; "
            f"either shrink the model or migrate to the XGBC cascade "
            f"format (u32 children) the scheduler uses"
        )
    for off in tree_roots:
        if off > XGB1_MAX_NODES:
            raise ValueError(
                f"tree root offset {off} exceeds u16 — see node-count error above"
            )

    # Pack XGB1 payload: header + roots + nodes.
    payload = bytearray()
    payload += XGB1_MAGIC
    payload += struct.pack(
        "<HHHHI",
        XGB1_PAYLOAD_VERSION_V1,
        0,                      # reserved u16
        n_trees,
        n_nodes,
        0,                      # reserved u32
    )
    for off in tree_roots:
        payload += struct.pack("<H", off)
    for (feat_idx, flags, left, right, threshold, value) in flat_nodes:
        payload += struct.pack(
            "<HHHHff",
            feat_idx, flags, left, right, threshold, value,
        )

    payload_bytes = bytes(payload)

    # Wrap in SEMB outer header.
    out = bytearray()
    out += SEMB_MAGIC
    out += struct.pack(
        "<HHHHI",
        SEMB_VERSION_V1,
        SEMB_KIND_XGBOOST,
        SEMB_FEATURE_SCHEMA_V1,
        0,                      # reserved u16
        len(payload_bytes),
    )
    out += struct.pack("<I", _fnv1a_32(payload_bytes))
    out += struct.pack("<I", 0)  # trailing reserved u32
    out += payload_bytes

    blob = bytes(out)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(blob)
    return blob


def generate_evict_verification_corpus(
    model: xgb.Booster,
    output_dir: str | Path,
    feature_names: list[str] | None = None,
    n_tests: int = 1000,
    seed: int = 54321,
) -> None:
    """Emit `test_vectors_xgb_evict.bin` (N × FEATURE_COUNT × f32) and
    `expected_evict.bin` (N × f32 sigmoid scores from
    `booster.predict()`). SLM-OS's `bench xgb-equiv-evict` replays the
    corpus through the blob-loaded predictor and asserts agreement
    within tolerance.

    Seed is fixed so re-running the exporter against the same trained
    model produces a bit-identical corpus — useful for CI diffing."""
    import numpy as np

    feature_names = feature_names or [
        f"f{i}" for i in range(EVICTION_FEATURE_COUNT)
    ]
    if len(feature_names) != EVICTION_FEATURE_COUNT:
        raise ValueError(
            f"feature_names length {len(feature_names)} != runtime "
            f"feature count {EVICTION_FEATURE_COUNT}"
        )
    rng = np.random.default_rng(seed)
    states = rng.random((n_tests, EVICTION_FEATURE_COUNT), dtype=np.float32)
    dmatrix = xgb.DMatrix(states, feature_names=feature_names)
    expected = model.predict(dmatrix).astype(np.float32)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    states.tofile(output_dir / "test_vectors_xgb_evict.bin")
    expected.tofile(output_dir / "expected_evict.bin")
