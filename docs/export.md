# Rust Export Pipeline

The export layer converts trained Python models to standalone Rust source code for integration into SLM-OS. Code lives in `src/export/`.

## Overview

SLM-OS runs on bare-metal ARM64 (Cortex-A78, Jetson Orin) with no Python runtime. Trained models must be compiled into the Rust binary as static code and const data. The export pipeline generates `.rs` files that can be dropped into `runtime/src/mm/` in the SLM-OS repo.

## XGBoost Export (`xgb_to_rust.py`)

Converts XGBoost's tree ensemble to a single Rust function with nested if-else chains.

### How it works

1. Dump the booster's trees to JSON via `booster.dump_model(dump_format='json')`
2. Walk each tree recursively, emitting `if features[split_idx] < threshold { ... } else { ... }`
3. Leaf nodes emit `return leaf_value;`
4. The top-level function sums outputs of all trees and applies sigmoid

### Generated code structure

```rust
// Auto-generated XGBoost decision tree — do not edit manually
pub fn xgb_predict(features: &[f32; 27]) -> f32 {
    let mut sum = 0.0_f32;
    sum += tree_0(features);
    sum += tree_1(features);
    // ... one function per tree
    1.0 / (1.0 + (-sum).exp())  // sigmoid
}

fn tree_0(f: &[f32; 27]) -> f32 {
    if f[3] < 42.5 {       // time_since_access
        if f[0] < 0.5 {    // recency_rank
            0.234
        } else {
            -0.112
        }
    } else {
        // ...
    }
}
```

### Size estimate

With depth-6 trees and 200 rounds: ~200 tree functions, each with up to 64 leaves. After Rust compilation, the resulting binary code is ~50KB.

## MLP Export (`mlp_to_rust.py`)

Converts MLP weights to Rust const arrays with optional int8 quantization.

### Quantization

Post-training symmetric per-tensor quantization:
- `scale = max(|weights|) / 127`
- `q = clamp(round(w / scale), -128, 127)`
- Each layer stores: `W_Ln: [[i8; in]; out]`, `B_Ln: [i8; out]`, `W_Ln_SCALE: f32`, `B_Ln_SCALE: f32`

### Int8 inference (default)

```rust
pub fn mlp_predict(features: &[f32; 27]) -> f32 {
    // Layer 1: 27 -> 64, ReLU
    let mut h1 = [0.0_f32; 64];
    for i in 0..64 {
        let mut acc = 0i32;
        for j in 0..27 {
            let input_q = (features[j] / W_L1_SCALE * 127.0) as i8;
            acc += (W_L1[i][j] as i32) * (input_q as i32);
        }
        h1[i] = (acc as f32) * W_L1_SCALE * W_L1_SCALE / (127.0 * 127.0)
            + (B_L1[i] as f32) * B_L1_SCALE;
        if h1[i] < 0.0 { h1[i] = 0.0; }  // ReLU
    }
    // ... layers 2, 3, output with sigmoid
}
```

### Float32 inference (verification)

A `mlp_predict_f32()` function using float32 weights is also generated. This matches the Python model exactly (no quantization error) and is used for cross-validation.

### Model sizes

| Variant | Size |
|---------|------|
| Int8 quantized | ~4 KB |
| Float32 | ~16 KB |

## Verification (`verify_export.py`)

Cross-validates Python model predictions against the exported Rust code:

```python
from src.export.verify_export import verify_predictions, generate_test_vectors

test_vectors = generate_test_vectors(num_samples=1000, num_features=27)
result = verify_predictions(python_probs, rust_probs, tolerance=0.01)
# result.max_error, result.mean_error, result.num_mismatches
```

`verify_eviction_decisions()` goes further: it groups by eviction_id and checks whether Python and Rust agree on which candidate to evict (argmax agreement), which is the metric that matters for correctness.

## CLI Usage

```bash
python scripts/export_to_slmos.py \
    --model-dir data/models/ \
    --output-dir data/export/ \
    --quantize  # default: int8 quantization enabled
```

Output files:
- `xgb_eviction.rs` -- XGBoost if-else chain
- `mlp_eviction_q8.rs` -- Int8 quantized MLP
- `mlp_eviction_f32.rs` -- Float32 MLP (for verification)
- `xgb_trees.json` -- Raw tree dump (for inspection)
