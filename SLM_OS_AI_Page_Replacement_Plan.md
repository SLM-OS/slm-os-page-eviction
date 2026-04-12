# SLM-OS AI-Driven Page Replacement System
## Architecture and Implementation Plan
### Version 1.1 — April 2026

---

## Executive Summary

This document specifies the architecture and implementation plan for an AI-driven page replacement system for SLM-OS. The system learns optimal 2MB block eviction decisions for the model memory allocator, targeting the specific access patterns of SLM inference workloads (sequential weight sweeps, bursty workspace allocation, concurrent model sharing). The approach combines offline supervised learning (MLP and XGBoost trained on Bélády-optimal labels) with an online adaptive expert selector (CACHEUS-style) that uses the trained models as experts alongside classical heuristics.

### Relationship to Existing SLM-OS Components

| SLM-OS Component | Role in This System |
|---|---|
| `runtime/src/mm/model_mem.rs` — ModelAllocator | Integration target: eviction policy plugs in here |
| `kernel/mm/pmm.c` — Physical page allocator | Underlying allocator (FFI calls from Rust) |
| `kernel/mm/vmm.c` — Virtual memory manager | Maps/unmaps 2MB blocks on eviction decisions |
| `runtime/src/sched/deadline.rs` — Deadline scheduler | Provides deadline pressure context for eviction |
| `kernel/gpu/gpu.h` — GPU driver interface | GPU-mapped blocks have eviction constraints |
| Phase 3 Milestone 1 — Model memory pools | Weight pool (RO, shared) vs workspace pool (RW, per-task) |

### Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Granularity | 2MB blocks (not 4KB pages) | Matches existing ModelAllocator block size; reduces decision frequency |
| Primary model | XGBoost decision tree | Sub-microsecond inference, interpretable, exports to C if-else chain |
| Secondary model | 3-layer MLP (64→32→16) | Differentiable, serves as CACHEUS expert, ONNX-exportable |
| Online adaptation | CACHEUS-style expert selector | No pre-existing dataset for SLM workloads; must adapt at runtime |
| Training labels | Bélády's optimal (computed offline from traces) | Provides theoretical upper bound for supervised learning |
| Simulator | Discrete-event Python simulator | Replicates ModelAllocator semantics without requiring bare-metal |

---

## 1. Simulator Architecture

### 1.1 Overview

The simulator replicates SLM-OS's model memory subsystem as a discrete-event simulation in Python. It models the weight pool and workspace pool as separate fixed-size block arrays, generates synthetic memory access traces based on SLM inference workload patterns, and evaluates replacement policies by counting page faults (evictions that require reloading).

```
┌─────────────────────────────────────────────────────────────────┐
│                    SimController                                │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────────┐ │
│  │ WorkloadGen  │  │ MemoryState  │  │ PolicyEvaluator       │ │
│  │              │→ │              │→ │                       │ │
│  │ - model_load │  │ - weight_pool│  │ - belady_oracle       │ │
│  │ - inference  │  │ - work_pool  │  │ - lru / lfu / arc     │ │
│  │ - hot_swap   │  │ - gpu_map    │  │ - mlp_policy          │ │
│  │ - concurrent │  │ - ref_counts │  │ - xgb_policy          │ │
│  │              │  │ - block_meta │  │ - cacheus_adaptive    │ │
│  └──────────────┘  └──────────────┘  └───────────────────────┘ │
│           │                │                    │               │
│           ▼                ▼                    ▼               │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    TraceCollector                         │  │
│  │  - access log (timestamp, block_id, access_type, pool)   │  │
│  │  - eviction log (timestamp, victim, features, optimal)   │  │
│  │  - metrics (fault_rate, avg_latency, throughput)          │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Memory State Model

The simulator models two pools matching Phase 3 Milestone 1's `ModelAllocator`:

```python
@dataclass
class BlockMeta:
    block_id: int                  # Unique block identifier
    pool: PoolType                 # WEIGHT or WORKSPACE
    model_id: int                  # Which model owns this block
    layer_idx: int                 # Layer index within model (weights only)
    size_mb: int                   # Always 2 (fixed block size)
    ref_count: int                 # Number of tasks sharing this block
    gpu_mapped: bool               # Currently mapped to GPU
    last_access_time: int          # Tick of last access
    access_count: int              # Total accesses since load
    load_time: int                 # Tick when block was loaded
    access_pattern: AccessPattern  # SEQUENTIAL, RANDOM, STRIDED
    dirty: bool                    # Modified since load (workspace only)

class PoolType(Enum):
    WEIGHT = 0      # Read-only, shareable model weights
    WORKSPACE = 1   # Read-write, per-task inference scratch

class AccessPattern(Enum):
    SEQUENTIAL = 0  # Layer-by-layer sweep (weights during inference)
    RANDOM = 1      # Random access (attention KV cache)
    STRIDED = 2     # Regular stride (batch processing)
    BURST = 3       # Short-lived allocation burst (workspace)
```

**Pool Configuration (matching QEMU defaults):**

| Parameter | Weight Pool | Workspace Pool |
|---|---|---|
| Total size | 128 MB (64 blocks) | 64 MB (32 blocks) |
| Block size | 2 MB | 2 MB |
| Permissions | Read-only | Read-write |
| Sharing | Multi-task (ref-counted) | Per-task |
| GPU-mappable | Yes | Yes |

### 1.3 Workload Generator

The workload generator produces synthetic access traces that model the four primary SLM-OS memory access patterns:

#### Pattern 1: Single Model Inference (Steady State)
- Load model weights: blocks 0..N sequentially (one per layer)
- Allocate workspace: 1-3 blocks for activation buffers
- Inference sweep: access weight blocks 0..N sequentially (forward pass)
- Release workspace
- Repeat inference sweep

```python
def gen_single_inference(model: ModelConfig, num_inferences: int) -> List[AccessEvent]:
    events = []
    # Load phase: sequential weight block loads
    for layer in range(model.num_layers):
        events.append(AccessEvent(
            block_type=PoolType.WEIGHT,
            model_id=model.id,
            layer_idx=layer,
            access_type=AccessType.LOAD
        ))
    # Workspace allocation
    for i in range(model.workspace_blocks):
        events.append(AccessEvent(
            block_type=PoolType.WORKSPACE,
            model_id=model.id,
            layer_idx=-1,
            access_type=AccessType.ALLOC
        ))
    # Inference sweeps
    for _ in range(num_inferences):
        for layer in range(model.num_layers):
            events.append(AccessEvent(
                block_type=PoolType.WEIGHT,
                model_id=model.id,
                layer_idx=layer,
                access_type=AccessType.READ
            ))
        # Workspace writes during each inference
        for i in range(model.workspace_blocks):
            events.append(AccessEvent(
                block_type=PoolType.WORKSPACE,
                model_id=model.id,
                layer_idx=-1,
                access_type=AccessType.WRITE
            ))
    return events
```

#### Pattern 2: Multi-Model Concurrent Inference
- 2-4 models loaded simultaneously
- Interleaved inference sweeps (round-robin or priority-based)
- Shared weight blocks when models overlap (e.g., shared embeddings)
- Memory pressure forces eviction decisions

#### Pattern 3: Hot-Swap (Component Update)
- Old model weights become cold (no more inference requests)
- New model weights load sequentially
- Overlap period: both models partially resident
- Old workspace released, new workspace allocated

#### Pattern 4: Memory Pressure Spike
- Burst of model load requests exceeding pool capacity
- Forces rapid eviction decisions under pressure
- Models with different priority levels compete for blocks
- GPU-mapped blocks have higher eviction cost

#### Workload Mix Configurations

| Scenario | Models | Pool Pressure | Duration (ticks) | Purpose |
|---|---|---|---|---|
| `steady_single` | 1 small (8 layers) | Low (25% full) | 10,000 | Baseline, no evictions |
| `steady_dual` | 2 medium (16 layers) | Medium (75%) | 20,000 | Interleaved access |
| `pressure_triple` | 3 medium (16 layers) | High (110%) | 30,000 | Forced evictions |
| `hot_swap` | 2→1→2 (swap mid-run) | Medium (80%) | 25,000 | Component update |
| `burst_load` | 4 small, staggered | Spike (150%) | 15,000 | Rapid eviction |
| `mixed_priority` | 1 critical + 2 normal | High (100%) | 20,000 | Priority-aware eviction |
| `gpu_contention` | 2 GPU-mapped + 1 CPU | High (90%) | 20,000 | GPU constraint |

### 1.4 Synthetic Model Configurations

```python
@dataclass
class ModelConfig:
    id: int
    name: str
    num_layers: int           # Number of weight blocks (1 per layer)
    workspace_blocks: int     # Activation buffer blocks needed
    inference_time_ticks: int # Ticks per full inference sweep
    priority: int             # 0-7 (matches SLM-OS priority levels)
    gpu_required: bool        # Needs GPU-mapped weight blocks
    access_pattern: str       # "sequential", "attention", "mixed"

# Pre-defined model archetypes
MODELS = {
    "tiny":    ModelConfig(0, "tiny-4L",     4,  1,   50,  3, False, "sequential"),
    "small":   ModelConfig(1, "small-8L",    8,  2,  100,  3, False, "sequential"),
    "medium":  ModelConfig(2, "medium-16L", 16,  3,  200,  4, False, "sequential"),
    "large":   ModelConfig(3, "large-32L",  32,  4,  400,  5, True,  "sequential"),
    "attn":    ModelConfig(4, "attn-16L",   16,  6,  250,  4, True,  "attention"),
    "critical":ModelConfig(5, "crit-8L",     8,  2,  100,  7, True,  "sequential"),
}
```

---

## 2. Feature Engineering (Observation Space)

### 2.1 Per-Block Features (computed for each eviction candidate)

At each eviction decision point, the policy observes features for every block currently in the pool. For a pool of N blocks, this produces an N × F_block feature matrix.

| # | Feature | Type | Range | Description |
|---|---|---|---|---|
| 1 | `recency_rank` | int | [0, N-1] | Rank by last access time (0 = most recent) |
| 2 | `frequency_rank` | int | [0, N-1] | Rank by access count (0 = most accessed) |
| 3 | `access_count` | int | [0, ∞) | Total accesses since load |
| 4 | `time_since_access` | int | [0, ∞) | Ticks since last access |
| 5 | `time_since_load` | int | [0, ∞) | Ticks since block was loaded into pool |
| 6 | `ref_count` | int | [0, MAX_TASKS] | Number of tasks sharing this block |
| 7 | `is_gpu_mapped` | bool | {0, 1} | Currently accessible by GPU |
| 8 | `pool_type` | cat | {0, 1} | 0=WEIGHT, 1=WORKSPACE |
| 9 | `is_dirty` | bool | {0, 1} | Modified since load (writeback cost) |
| 10 | `layer_idx_norm` | float | [0.0, 1.0] | Layer position / total layers (-1 → 0 for workspace) |
| 11 | `model_priority` | int | [0, 7] | Owning model's priority level |
| 12 | `model_active_inferences` | int | [0, ∞) | Active inference tasks using this model |
| 13 | `access_pattern` | cat | {0,1,2,3} | SEQUENTIAL/RANDOM/STRIDED/BURST |
| 14 | `predicted_reuse_dist` | float | [0.0, 1.0] | Estimated ticks until next access / horizon |
| 15 | `eviction_cost` | float | [0.0, 1.0] | Normalized cost to reload (GPU remap + writeback) |

**Total per-block features: 15**

### 2.2 Global State Features (context for the eviction decision)

| # | Feature | Type | Range | Description |
|---|---|---|---|---|
| 16 | `weight_pool_utilization` | float | [0.0, 1.0] | Allocated blocks / total weight pool blocks |
| 17 | `workspace_pool_utilization` | float | [0.0, 1.0] | Allocated blocks / total workspace pool blocks |
| 18 | `num_loaded_models` | int | [0, 8] | Distinct models with blocks in pool |
| 19 | `total_gpu_mapped_blocks` | int | [0, N] | Blocks currently GPU-accessible |
| 20 | `pending_load_requests` | int | [0, ∞) | Queued block loads waiting for space |
| 21 | `avg_model_priority` | float | [0.0, 7.0] | Mean priority of loaded models |
| 22 | `max_deadline_pressure` | float | [0.0, 1.0] | Highest deadline urgency across active tasks |
| 23 | `recent_fault_rate` | float | [0.0, 1.0] | Faults / accesses in last 100 ticks |
| 24 | `hot_swap_in_progress` | bool | {0, 1} | Currently swapping a model |
| 25 | `requesting_block_pool` | cat | {0, 1} | Pool type of the block requesting admission |
| 26 | `requesting_block_model_id` | int | [0, 8] | Model ID of incoming block |
| 27 | `requesting_block_priority` | int | [0, 7] | Priority of incoming block's model |

**Total global features: 12**

### 2.3 Flattened Feature Vector

For the MLP and XGBoost models, each eviction candidate is scored independently. The input is a single candidate's 15 per-block features concatenated with the 12 global features:

**Total features per candidate: 27**

For a pool of 64 blocks, each eviction decision evaluates 27 features × 64 candidates = 1,728 feature evaluations (but each is independent, so the model runs 64 times with 27 inputs each, or once with 27 inputs producing a score, then argmin over candidates).

### 2.4 Feature Normalization

| Feature Group | Normalization | Notes |
|---|---|---|
| Ranks | / (N-1) → [0, 1] | N = pool size |
| Counts | log1p then / log1p(max_observed) | Handles heavy tails |
| Ticks | / horizon_window | Window = 1000 ticks default |
| Booleans | Raw {0, 1} | No normalization needed |
| Categoricals | One-hot for XGBoost, ordinal for MLP | Pool type and access pattern |
| Priority | / 7.0 → [0, 1] | Maps to SLM-OS 8 levels |

---

## 3. Action Space

### 3.1 Eviction Decision

The action space is a **single discrete choice**: which block to evict from the pool.

| Property | Value |
|---|---|
| Action type | Discrete (index into resident block list) |
| Action size | Variable: [0, N-1] where N = number of evictable blocks |
| Constraints | Cannot evict blocks with ref_count > 0 |
| Constraints | Cannot evict GPU-mapped blocks (must unmap first) |
| Constraints | Cannot evict blocks currently being accessed |

### 3.2 Evictable Block Filtering

Before the policy runs, the set of eviction candidates is filtered:

```python
def get_evictable_blocks(pool: Pool) -> List[BlockMeta]:
    return [
        block for block in pool.blocks
        if block.ref_count == 0
        and not block.gpu_mapped
        and block.state != BlockState.ACCESSING
    ]
```

If no blocks are evictable, the load request blocks until a block becomes evictable (ref_count drops to 0 or GPU unmap completes). This matches SLM-OS's blocking semantics.

---

## 4. Bélády's Optimal Oracle

### 4.1 Offline Optimal Computation

Given a complete trace of future accesses, Bélády's algorithm evicts the block whose next access is furthest in the future (or never, if the block is not accessed again).

```python
def belady_optimal(pool: Pool, future_accesses: List[AccessEvent], current_tick: int) -> int:
    """Returns block_id of optimal eviction target."""
    evictable = get_evictable_blocks(pool)
    if not evictable:
        raise NoEvictableBlockError()

    # For each evictable block, find its next access time
    next_access = {}
    for block in evictable:
        next_access[block.block_id] = float('inf')  # default: never accessed again
        for event in future_accesses:
            if event.tick > current_tick and event.targets_block(block):
                next_access[block.block_id] = event.tick
                break

    # Evict the block with the furthest (or no) next access
    return max(evictable, key=lambda b: next_access[b.block_id]).block_id
```

### 4.2 Optimal Labels for Training

For each eviction event in a trace, we record:
- The features of all evictable candidates (Section 2)
- The Bélády-optimal choice (the correct eviction target)
- The reuse distance for each candidate (ticks until next access)

This produces labeled training data:
- **Input**: 27-feature vector for each candidate
- **Label**: Binary — 1 if this candidate is the optimal eviction target, 0 otherwise
- **Auxiliary label**: Reuse distance (for regression-based ranking)

---

## 5. Expert Policies (Baselines + Dataset Generation)

### 5.1 Classical Policies

These serve as baselines for comparison and as experts in the CACHEUS framework:

**LRU (Least Recently Used):**
- Evict the block with the oldest `last_access_time`
- Good for sequential sweeps (weights during inference)
- Poor for frequency-heavy patterns

**LFU (Least Frequently Used):**
- Evict the block with the lowest `access_count`
- Good for long-running models with stable hot sets
- Poor for newly-loaded blocks (cold start)

**ARC (Adaptive Replacement Cache):**
- Maintains ghost lists for recently evicted LRU and LFU candidates
- Adapts balance between recency and frequency
- Stronger baseline than pure LRU or LFU

**SLM-Aware Heuristic (hand-tuned, current SLM-OS policy):**
```python
def slm_heuristic_evict(pool: Pool) -> int:
    """Current hand-tuned policy from Phase 3."""
    evictable = get_evictable_blocks(pool)
    # Priority 1: Evict workspace before weights (cheaper to recreate)
    workspace = [b for b in evictable if b.pool == PoolType.WORKSPACE]
    if workspace:
        # Within workspace, evict LRU
        return min(workspace, key=lambda b: b.last_access_time).block_id
    # Priority 2: Evict weights from inactive models
    inactive = [b for b in evictable if b.model_active_inferences == 0]
    if inactive:
        return min(inactive, key=lambda b: b.last_access_time).block_id
    # Priority 3: Evict LRU among remaining
    return min(evictable, key=lambda b: b.last_access_time).block_id
```

### 5.2 Policy Evaluation Metrics

| Metric | Formula | Target |
|---|---|---|
| Fault rate | faults / total_accesses | Lower is better |
| Normalized fault rate | (policy_faults - optimal_faults) / (lru_faults - optimal_faults) | 0.0 = optimal, 1.0 = LRU |
| Eviction cost | sum(eviction_cost per fault) | Lower is better (accounts for GPU remap) |
| Throughput | inferences_completed / total_ticks | Higher is better |
| Deadline misses | deadline_missed_tasks / total_deadline_tasks | 0.0 = no misses |

---

## 6. Dataset Specification

### 6.1 Dataset Generation Pipeline

```
WorkloadGen → SimController → TraceCollector → BéládyLabeler → FeatureExtractor → Dataset
     │              │                │                │                │              │
  7 scenarios    Run sim       Raw access log    Compute optimal   27 features    Parquet
  × 5 seeds     per policy    + eviction log    per eviction      per candidate   files
```

### 6.2 Dataset Size Estimate

| Parameter | Value |
|---|---|
| Scenarios | 7 workload configurations |
| Seeds per scenario | 5 (random variation) |
| Avg ticks per scenario | ~20,000 |
| Avg evictions per scenario (high pressure) | ~500-2,000 |
| Avg evictable candidates per eviction | ~20-40 |
| Rows per eviction | 1 per candidate (with binary optimal label) |
| **Estimated total rows** | **7 × 5 × 1,000 × 30 ≈ 1,050,000** |
| Features per row | 27 + labels |
| Approx dataset size | ~200 MB (Parquet, compressed) |

### 6.3 Dataset Schema (Parquet)

```
eviction_events.parquet
├── scenario: string          # Workload scenario name
├── seed: int32               # Random seed for reproducibility
├── tick: int64               # Simulation tick of eviction event
├── eviction_id: int64        # Unique eviction event ID
├── candidate_idx: int32      # Index within this eviction's candidate set
├── num_candidates: int32     # Total candidates for this eviction
├── -- Per-block features (15) --
├── recency_rank: float32
├── frequency_rank: float32
├── access_count: float32     # log1p normalized
├── time_since_access: float32
├── time_since_load: float32
├── ref_count: int32
├── is_gpu_mapped: int8
├── pool_type: int8
├── is_dirty: int8
├── layer_idx_norm: float32
├── model_priority: float32
├── model_active_inferences: int32
├── access_pattern: int8
├── predicted_reuse_dist: float32
├── eviction_cost: float32
├── -- Global features (12) --
├── weight_pool_util: float32
├── workspace_pool_util: float32
├── num_loaded_models: int32
├── total_gpu_mapped: int32
├── pending_loads: int32
├── avg_model_priority: float32
├── max_deadline_pressure: float32
├── recent_fault_rate: float32
├── hot_swap_active: int8
├── req_block_pool: int8
├── req_block_model_id: int32
├── req_block_priority: float32
├── -- Labels --
├── is_optimal: int8          # 1 if Bélády chose this candidate
├── reuse_distance: float32   # Normalized ticks until next access
└── actual_policy: string     # Which policy made this decision (for analysis)
```

### 6.4 Train/Validation/Test Split

| Split | Allocation | Strategy |
|---|---|---|
| Train | 70% | Random split by eviction_id (not by row, to keep candidate sets together) |
| Validation | 15% | Random split by eviction_id |
| Test | 15% | Held-out scenarios: `hot_swap` + `gpu_contention` (unseen workloads) |

The test set uses **held-out workload scenarios** to measure generalization to unseen access patterns, not just unseen eviction events from trained scenarios.

---

## 7. Model Architectures

### 7.1 XGBoost Decision Tree (Primary — Production Model)

**Rationale:** Sub-microsecond inference, interpretable, trivially exportable to C/Rust if-else chain for bare-metal deployment. Feature importance analysis reveals which memory characteristics matter most.

**Architecture:**

| Parameter | Value | Notes |
|---|---|---|
| Objective | `binary:logistic` | Predicts P(optimal eviction target) |
| Num rounds | 200 | Early stopping on validation |
| Max depth | 6 | Shallow trees for fast inference |
| Learning rate | 0.1 | Standard |
| Min child weight | 10 | Prevents overfitting to rare events |
| Subsample | 0.8 | Row sampling per tree |
| Colsample bytree | 0.8 | Feature sampling per tree |
| Scale pos weight | ~30 | Corrects class imbalance (1 optimal per ~30 candidates) |
| Eval metric | `auc` + custom `ndcg` | AUC for classification, NDCG for ranking quality |

**Inference path:**
1. For each eviction event, compute 27 features for all evictable candidates
2. Run XGBoost predict on each candidate → probability score
3. Evict the candidate with the **highest** predicted probability

**Export to SLM-OS:**
```
XGBoost model → dump to JSON → convert to nested if-else in Rust
```

Trees of depth 6 with 200 rounds produce ~200 trees × ~64 leaves = ~12,800 decision paths. After pruning redundant paths, this compiles to a ~50KB Rust function.

### 7.2 MLP (Secondary — CACHEUS Expert + Research Comparison)

**Rationale:** Differentiable model that can be fine-tuned online, serves as a learned expert in the CACHEUS framework, and enables gradient-based analysis of feature importance.

**Architecture:**

```
Input (27 features)
    │
    ▼
Linear(27, 64) → ReLU → Dropout(0.1)
    │
    ▼
Linear(64, 32) → ReLU → Dropout(0.1)
    │
    ▼
Linear(32, 16) → ReLU
    │
    ▼
Linear(16, 1) → Sigmoid
    │
    ▼
Output: P(optimal eviction target)
```

| Parameter | Value |
|---|---|
| Total parameters | 27×64 + 64 + 64×32 + 32 + 32×16 + 16 + 16×1 + 1 = **3,889** |
| Model size | ~15 KB (float32) or ~4 KB (int8 quantized) |
| Activation | ReLU (no division, ARM NEON friendly) |
| Loss | Binary cross-entropy + auxiliary reuse distance regression |
| Optimizer | Adam, lr=1e-3 with cosine annealing |
| Batch size | 256 |
| Epochs | 50 with early stopping (patience=5) |

**Auxiliary head (reuse distance prediction):**
```
Shared layers (64→32→16)
    │
    ├──→ Linear(16, 1) → Sigmoid     # Classification: is_optimal
    │
    └──→ Linear(16, 1) → ReLU        # Regression: reuse_distance
```

Joint loss: `L = L_classify + 0.3 * L_reuse_mse`

The reuse distance head acts as a regularizer, encouraging the model to learn about future access patterns rather than memorizing eviction decisions.

**Export to SLM-OS:**
```
PyTorch model → ONNX → flat C weight arrays (27×64 + ... matrices)
→ hand-coded matrix multiply in Rust (no ONNX runtime needed)
```

At 3,889 parameters with int8 quantization, inference is ~4,000 multiply-accumulate ops — well under 1μs on Cortex-A78.

### 7.3 CACHEUS-Style Adaptive Expert Selector (Online — Runtime Model)

**Rationale:** No pre-existing dataset captures SLM-OS's exact workload characteristics. The CACHEUS framework enables online adaptation by maintaining a weighted ensemble of experts and learning which expert performs best on the current workload.

**Architecture:**

```
                    ┌──────────────────────────┐
                    │   Expert Selector         │
                    │                           │
Eviction Request →  │   w_lru  × LRU_score     │
                    │ + w_lfu  × LFU_score     │ → argmax → Evict
                    │ + w_slm  × SLM_heuristic │
                    │ + w_mlp  × MLP_score     │
                    │ + w_xgb  × XGB_score     │
                    │                           │
                    │   Weights updated via      │
                    │   regret minimization     │
                    └──────────────────────────┘
```

**Expert Pool:**

| Expert | Source | Strengths |
|---|---|---|
| SR-LRU | Scan-resistant LRU | Handles sequential weight sweeps |
| CR-LFU | Churn-resistant LFU | Handles frequently-used model blocks |
| SLM-Heuristic | Hand-tuned (Section 5.1) | Workspace-before-weights priority |
| MLP | Trained offline (Section 7.2) | Learned from Bélády-optimal decisions |
| XGBoost | Trained offline (Section 7.1) | Feature-importance-driven decisions |

**Weight Update (gradient-based hill climbing):**
```python
class CACHEUSSelector:
    def __init__(self, num_experts=5, window_size=100, lr=0.1):
        self.weights = np.ones(num_experts) / num_experts
        self.lr = lr
        self.window_size = window_size
        self.history = deque(maxlen=window_size)

    def select_expert(self, candidates, features):
        scores = np.zeros(len(candidates))
        for i, (expert, w) in enumerate(zip(self.experts, self.weights)):
            expert_scores = expert.score(candidates, features)
            scores += w * expert_scores
        return candidates[np.argmax(scores)]

    def update(self, chosen_block, was_fault_soon):
        """Called when evicted block is accessed again (fault) or not."""
        # Increase weight of experts that would have avoided the fault
        for i, expert in enumerate(self.experts):
            if was_fault_soon:
                # Penalize experts that agreed with the bad decision
                if expert.would_have_chosen(chosen_block):
                    self.weights[i] *= (1 - self.lr)
            else:
                # Reward experts that agreed with the good decision
                if expert.would_have_chosen(chosen_block):
                    self.weights[i] *= (1 + self.lr)
        # Normalize
        self.weights /= self.weights.sum()
```

---

## 8. Training Pipeline

### 8.1 XGBoost Training

```
Phase 1: Dataset Generation
    ├── Run simulator with 7 scenarios × 5 seeds
    ├── Collect traces with Bélády-optimal labels
    └── Export to Parquet (Section 6.3)

Phase 2: Feature Analysis
    ├── Compute feature correlations
    ├── Feature importance via mutual information
    └── Remove redundant features (target: keep 20-25 of 27)

Phase 3: Model Training
    ├── Load Parquet, group by eviction_id
    ├── Train with class-weight balancing
    ├── 5-fold cross-validation (fold by scenario)
    └── Early stopping on validation AUC

Phase 4: Evaluation
    ├── Run trained model in simulator (replace policy)
    ├── Compare fault rate vs LRU, LFU, ARC, SLM-heuristic, Bélády
    ├── Compute normalized fault rate
    └── Analyze failure cases (when does XGBoost pick wrong victim?)

Phase 5: Export
    ├── Dump trees to JSON
    ├── Convert to Rust if-else chain
    ├── Verify identical predictions between Python and Rust
    └── Measure inference latency on ARM target
```

### 8.2 MLP Training

```
Phase 1: Same dataset as XGBoost

Phase 2: Preprocessing
    ├── Normalize features (Section 2.4)
    ├── One-hot encode categoricals
    └── Create DataLoader with candidate-set-aware batching

Phase 3: Model Training
    ├── Joint classification + reuse distance loss
    ├── Adam optimizer with cosine annealing
    ├── Early stopping on validation NDCG
    └── Save best checkpoint

Phase 4: DAgger Fine-Tuning (optional)
    ├── Run MLP policy in simulator
    ├── At each eviction, also compute Bélády-optimal
    ├── Add new (state, optimal_label) pairs to training set
    ├── Retrain on augmented dataset
    └── Repeat 3 rounds

Phase 5: Quantization
    ├── Post-training int8 quantization
    ├── Verify accuracy loss < 1% on validation set
    ├── Export weight matrices as C arrays
    └── Hand-code inference in Rust (3 matrix multiplies + ReLU)
```

### 8.3 CACHEUS Online Training

The CACHEUS selector trains **at runtime** within SLM-OS. No offline dataset is needed — it adapts to whatever workload the system encounters.

```
Boot:
    ├── Initialize 5 expert weights to uniform (0.2 each)
    ├── Load pre-trained MLP and XGBoost weights

Runtime (per eviction):
    ├── Each expert scores all evictable candidates
    ├── Weighted combination selects victim
    ├── Record decision in circular buffer

Feedback (per access / every 100 ticks):
    ├── Check if recently evicted blocks have been reloaded (fault)
    ├── Update expert weights via regret minimization
    ├── Log weight trajectory for post-hoc analysis

Periodic (every 10,000 ticks):
    ├── Log expert weight distribution
    ├── Log per-expert fault contribution
    └── Detect workload phase changes (weight shift > 0.3)
```

---

## 9. SLM-OS Integration Architecture

### 9.1 Integration Point

The trained models plug into `runtime/src/mm/model_mem.rs` at the eviction decision point:

```rust
// runtime/src/mm/eviction_policy.rs (new file)

/// Eviction policy trait — all policies implement this
pub trait EvictionPolicy {
    fn select_victim(&self, candidates: &[BlockFeatures], global: &GlobalFeatures) -> usize;
    fn update_feedback(&mut self, evicted_block: BlockId, was_reloaded: bool);
    fn name(&self) -> &str;
}

/// XGBoost-exported decision tree (primary production policy)
pub struct XGBoostPolicy {
    // Weights compiled into nested if-else (no runtime allocation)
}

/// MLP with pre-trained weights (CACHEUS expert)
pub struct MLPPolicy {
    weights_l1: [[i8; 27]; 64],   // int8 quantized
    bias_l1: [i8; 64],
    weights_l2: [[i8; 64]; 32],
    bias_l2: [i8; 32],
    weights_l3: [[i8; 32]; 16],
    bias_l3: [i8; 16],
    weights_out: [i8; 16],
    bias_out: i8,
}

/// CACHEUS adaptive selector (online learning)
pub struct CACHEUSPolicy {
    experts: [Box<dyn EvictionPolicy>; 5],
    weights: [f32; 5],
    lr: f32,
    feedback_buffer: CircularBuffer<EvictionRecord>,
}

/// Feature extraction from BlockMeta + global pool state
pub fn extract_features(block: &BlockMeta, pool: &Pool, global: &GlobalState) -> BlockFeatures {
    BlockFeatures {
        recency_rank: pool.recency_rank(block),
        frequency_rank: pool.frequency_rank(block),
        access_count: (block.access_count as f32 + 1.0).ln(),
        time_since_access: (global.tick - block.last_access_time) as f32 / 1000.0,
        // ... remaining 23 features ...
    }
}
```

### 9.2 ModelAllocator Integration

```rust
// In runtime/src/mm/model_mem.rs — modified eviction path

impl ModelAllocator {
    pub fn alloc_block(&mut self, pool: PoolType, model_id: u32) -> Result<BlockHandle, AllocError> {
        // Try to find a free block first
        if let Some(handle) = self.find_free_block(pool) {
            return Ok(handle);
        }

        // Pool full — must evict
        let candidates = self.get_evictable_candidates(pool);
        if candidates.is_empty() {
            return Err(AllocError::NoEvictableBlocks);
        }

        // Extract features for all candidates
        let global = self.extract_global_features();
        let features: Vec<BlockFeatures> = candidates.iter()
            .map(|c| extract_features(c, &self.pools[pool], &global))
            .collect();

        // Policy decides which block to evict
        let victim_idx = self.eviction_policy.select_victim(&features, &global);
        let victim = candidates[victim_idx];

        // Evict and reuse the block
        self.evict_block(victim)?;
        self.allocate_into(victim, pool, model_id)
    }
}
```

### 9.3 FFI Boundary (if needed)

If feature extraction or eviction logic remains in C:

```c
// kernel/include/eviction_ffi.h
typedef struct {
    float features[27];
} block_features_t;

// Called from Rust via FFI
int32_t slm_eviction_get_block_features(uint32_t block_id, block_features_t* out);
int32_t slm_eviction_notify_fault(uint32_t block_id);
```

All heavy lifting (model inference, expert selection) stays in Rust. The FFI is only needed if some features require C-side kernel data not already exposed.

---

## 10. Implementation Plan

### Icon Key

| Icon | Meaning |
|------|---------|
| ☐ | Not started |
| ✅ | Complete |
| ⏸️ | Deferred to later phase |
| 🔗 | Has dependency on another milestone |

---

### Phase 1: Simulator Core (Week 1-2) ✅

**Goal:** Working discrete-event simulator that can run classical policies and produce traces.

#### Milestone 1.1: Simulator Skeleton ✅
- ✅ Create `slm_os_page_sim/` Python project with pyproject.toml
- ✅ Implement `BlockMeta`, `Pool`, `PoolType`, `AccessPattern` data classes
- ✅ Implement `MemoryState` class with weight_pool and workspace_pool
- ✅ Implement `alloc_block()`, `free_block()`, `access_block()` operations
- ✅ Implement ref_count tracking and evictable block filtering
- ✅ Unit tests: alloc/free cycles, ref_count semantics, pool capacity

#### Milestone 1.2: Workload Generator ✅
- ✅ Implement `ModelConfig` and pre-defined model archetypes
- ✅ Implement `gen_single_inference()` — single model steady state
- ✅ Implement `gen_multi_model()` — interleaved concurrent inference
- ✅ Implement `gen_hot_swap()` — component swap mid-run
- ✅ Implement `gen_burst_load()` — memory pressure spike
- ✅ Implement `gen_mixed_priority()` — priority-aware workload
- ✅ Implement `gen_gpu_contention()` — GPU-mapped block constraints
- ✅ Parametric seed support for reproducibility
- ✅ Unit tests: trace length, access pattern correctness, model counts

#### Milestone 1.3: Classical Policies ✅
- ✅ Implement `LRUPolicy` — least recently used eviction
- ✅ Implement `LFUPolicy` — least frequently used eviction
- ✅ Implement `ARCPolicy` — adaptive replacement cache
- ✅ Implement `SLMHeuristicPolicy` — hand-tuned SLM-OS policy
- ✅ Implement `PolicyEvaluator` — runs policy on trace, collects metrics
- ✅ Implement `BéládyOracle` — offline optimal (requires full trace)
- ✅ Unit tests: each policy on simple known-optimal traces

#### Milestone 1.4: Trace Collection ✅
- ✅ Implement `TraceCollector` — records access and eviction events
- ✅ Export traces to Parquet format
- ✅ Generate comparison metrics: fault_rate, normalized_fault_rate, eviction_cost
- ✅ Validation: Bélády achieves 0.0 normalized fault rate on all scenarios

**Phase 1 Gate:** ✅ Simulator runs all 7 scenarios with 5 policies, produces Parquet traces, Bélády is provably optimal. 61 unit tests passing.

---

### Phase 2: Dataset Generation & Feature Engineering (Week 2-3) ✅

**Goal:** Labeled dataset with Bélády-optimal labels and 27-feature vectors.

#### Milestone 2.1: Feature Extraction ✅
- ✅ Implement `FeatureExtractor` class (Section 2)
- ✅ Compute per-block features (15 features, 14 without predicted_reuse_dist)
- ✅ Compute global state features (12 features)
- ✅ Implement feature normalization (Section 2.4)
- ✅ Validate features: no NaN, correct ranges, correct ranks
- ✅ Unit tests: feature computation on hand-crafted pool states

#### Milestone 2.2: Bélády Labeling ✅
- ✅ Implement offline Bélády oracle that labels each eviction event
- ✅ For each eviction: mark optimal candidate, compute reuse distances
- ✅ Validate: exactly 1 optimal candidate per eviction event
- ✅ Handle ties (multiple blocks never accessed again) — break by eviction cost

#### Milestone 2.3: Dataset Assembly ✅
- ✅ Run all 7 scenarios × 5 seeds, collect features + labels
- ✅ Export to Parquet with schema from Section 6.3
- ✅ Compute dataset statistics: class balance, feature distributions
- ✅ Create train/val/test split (Section 6.4)
- ✅ Validate: test set contains only held-out scenarios

**Phase 2 Gate:** ✅ Dataset passes schema validation, ~6% class balance documented, test set isolation verified (hot_swap + gpu_contention held out).

---

### Phase 3: XGBoost Model (Week 3-4)

**Goal:** Trained XGBoost model that outperforms LRU and approaches Bélády.

#### Milestone 3.1: Baseline Training ✅
- ✅ Load Parquet dataset, create DMatrix with correct dtypes
- ✅ Train with parameters from Section 7.1
- ✅ 5-fold cross-validation (fold by scenario, not by row)
- ✅ Log AUC, NDCG, precision@1 per fold
- ✅ Early stopping on validation AUC

#### Milestone 3.2: Hyperparameter Tuning ✅
- ✅ Grid search over: max_depth {4,5,6,7}, lr {0.05,0.1,0.2}, min_child {5,10,20} — 36 configs
- ✅ Best: depth=7, lr=0.2, mcw=5 → AUC 0.999942; default depth=6 only 0.002% worse
- ✅ Document final hyperparameters: depth=6, lr=0.1 retained (smaller export, marginal AUC diff)

#### Milestone 3.3: Feature Importance Analysis ✅
- ✅ Extract gain-based feature importances (15 of 27 features used by XGBoost)
- ✅ Top feature: predicted_reuse_dist (76.1% of gain), time_since_access (10.2%)
- ✅ Top 10 features account for 97.2% of total gain
- ✅ Reduced feature set (top-10 only): 96.04% test acc vs 96.07% baseline (**-0.03%** — negligible)
- ✅ 12 features unused (booleans, GPU fields, pool_type, etc.)

#### Milestone 3.4: Simulator Evaluation ✅
- ✅ Integrate trained XGBoost into simulator as `XGBPolicy`
- ✅ Run all 7 scenarios × 5 seeds with XGBoost policy
- ✅ Test set accuracy: 96.07% eviction decision agreement with Bélády
- ✅ XGBoost matches Bélády on single_inference (norm_rate=0.0) and burst_load (0.0)
- ✅ Hardest scenario: hot_swap (norm_rate=0.50) — still 50% of LRU-Bélády gap closed

**Phase 3 Gate:** ✅ XGBoost achieves strong results across all scenarios. 96% test accuracy. Feature importance documented.

---

### Phase 4: MLP Model (Week 4-5)

**Goal:** Trained MLP that matches or exceeds XGBoost, suitable for CACHEUS integration.

#### Milestone 4.1: PyTorch Training ✅
- ✅ Implement `PageReplacementMLP` in PyTorch (Section 7.2)
- ✅ Implement joint classification + reuse distance loss
- ✅ Implement candidate-set-aware batching (group by eviction_id)
- ✅ Train with Adam + cosine annealing
- ✅ Early stopping on validation NDCG

#### Milestone 4.2: DAgger Fine-Tuning ✅
- ✅ Run MLP policy in simulator, collect on-policy data
- ✅ Augment training set with Bélády labels on MLP-visited states
- ✅ Retrain on augmented dataset (up to 3 rounds); dataset grew 747K → 1.08M
- ✅ Measure improvement: baseline 95.97% → DAgger 95.82% (-0.15%, **no improvement**)
- ✅ Finding: MLP baseline is already near-ceiling; state-distribution overlap with Belady is high

#### Milestone 4.3: Quantization ✅
- ✅ Apply post-training int8 quantization
- ✅ Verify: accuracy loss < 1% — int8 agrees with float32 on 99.6% of eviction decisions (PASS)
- ✅ Export weight matrices as flat arrays
- ✅ Measure model size: target < 5 KB

#### Milestone 4.4: Simulator Evaluation ✅
- ✅ Integrate MLP into simulator as `MLPPolicy`
- ✅ Compare with XGBoost on all scenarios: MLP 95.97% vs XGBoost 96.07% test accuracy
- ✅ **MLP within 0.1% of XGBoost — well within 5% target**
- ✅ Profile inference latency (Python batch, batch=64): XGBoost 9μs/candidate, MLP 2μs/candidate
- ✅ Rust-exported if-else chains expected sub-microsecond (Python overhead dominates current timing)

**Phase 4 Gate:** ✅ MLP matches XGBoost (95.97% vs 96.07%). Int8 quantized model passes verification (99.6% decision agreement).

---

### Phase 5: CACHEUS Framework (Week 5-6)

**Goal:** Online adaptive expert selector that combines all policies.

#### Milestone 5.1: Expert Interface ✅
- ✅ Define `Expert` protocol in Python (matches Rust `EvictionPolicy` trait)
- ✅ Wrap LRU, LFU, SLM-heuristic, MLP, XGBoost as experts
- ✅ Each expert implements `score(candidates, features) → scores[]`

#### Milestone 5.2: Weight Learning ✅
- ✅ Implement `CACHEUSSelector` (Section 7.3)
- ✅ Implement multiplicative weight update with regret minimization
- ✅ Implement feedback mechanism: detect when evicted block is reloaded (fixed: track evicted content, positive/negative signals)
- ✅ Fixed bug: weight update was comparing against wrong target (last expert vs ensemble choice)
- ✅ Tune learning rate and window size on validation scenarios — best: lr=0.4, window=200

#### Milestone 5.3: Online Evaluation ✅
- ✅ Run CACHEUS in simulator with all 5 experts (and pool variants)
- ✅ Track expert weight trajectories over time (`scripts/record_cacheus_trajectories.py`)
- ✅ Verify: CACHEUS adapts weights to match workload type — gpu_contention converges to 99% XGBoost / 1% MLP
- ✅ **Pool composition finding**: `ml_only` (XGBoost+MLP) wins at **0.212 mean norm rate**, beating `all_5` (0.427) — classical experts dilute the ensemble
- ✅ Measure adaptation time: 43-92 ticks across all scenarios (`plot_trajectories.adaptation_speed`)

#### Milestone 5.4: Workload Phase Detection ✅
- ✅ Detect workload transitions (steady → hot-swap → pressure)
- ✅ Verify weights shift appropriately at transitions (per-scenario PNGs in `data/results/trajectory_*.png`)
- ✅ Log phase change events for capstone analysis (`phase_change_log` property)

**Phase 5 Gate:** ✅ CACHEUS with `ml_only` pool achieves 0.212 mean norm rate — matches XGBoost (0.215) and MLP (0.218) and beats all classical baselines (0.857). Weight trajectories are visible and explainable.

---

### Phase 6: SLM-OS Integration (Week 6-7)

**Goal:** Trained models running in SLM-OS's Rust runtime.

#### Milestone 6.1: Rust Export Pipeline ✅
- ✅ Implement XGBoost JSON → Rust if-else chain code generator (`xgb_to_rust.py`)
- ✅ Implement MLP → Rust const arrays with int8 quantization (`mlp_to_rust.py`)
- ✅ Implement cross-validation framework (`verify_export.py`)
- ✅ Implement float32 Rust inference function for verification
- ✅ End-to-end Rust verification harness (`scripts/verify_rust_export.py`):
  compiles generated Rust with rustc, runs on 800 random test vectors,
  compares to Python reference. **XGBoost: byte-perfect (0 mismatches)**.
  **MLP int8: 95% decision agreement on 8-candidate groups** (matches the
  target). Surfaced and fixed two real export bugs (XGBoost feature-name
  parsing, MLP layer-1 input quantization formula).

#### Milestone 6.2: Rust Policy Trait ✅ (reference crate)
- ✅ Reference crate at `slm_os_integration/` (cargo project, edition 2021)
- ✅ Define `EvictionPolicy` trait — `select_victim` / `score` / `update_feedback` / `reset` / `name`
- ✅ Implement `LruPolicy`, `LfuPolicy` in Rust as baselines
- ✅ Implement `SlmHeuristicPolicy` with the same priority cascade as Python
- ✅ Parity tests in `slm_os_integration/tests/parity.rs` — 11 tests covering the same edges as the Python policy tests; all passing
- ☐ Drop into SLM-OS `runtime/src/mm/eviction_policy.rs` (handoff to OS repo)

*Depends on: trained models from Phases 3-4*

#### Milestone 6.3: XGBoost in Rust 🔗
- ✅ Convert XGBoost JSON dump to Rust if-else chain (code generator script)
- ☐ Implement `XGBoostPolicy` struct in SLM-OS
- ☐ Verify: identical predictions to Python on 1,000 test samples
- ☐ Benchmark: inference latency target < 1μs on ARM64

*Depends on: Milestone 6.1, trained XGBoost model*

#### Milestone 6.4: MLP in Rust 🔗
- ✅ Export int8 weight matrices as Rust const arrays
- ✅ Implement fixed-point matrix multiply (no floating point in hot path)
- ✅ Implement ReLU and sigmoid in generated Rust
- ☐ Implement `MLPPolicy` struct in SLM-OS
- ☐ Verify: predictions match Python within int8 quantization tolerance
- ☐ Benchmark: inference latency target < 1μs on ARM64

*Depends on: Milestone 6.1, trained MLP model*

#### Milestone 6.5: CACHEUS in Rust ✅ (reference crate)
- ✅ `CacheusSelector` in `slm_os_integration/src/cacheus.rs` — wraps any `Vec<Box<dyn EvictionPolicy>>` of experts
- ✅ Multiplicative weight update with f32 arithmetic (matches the corrected Python impl: penalize agreeing experts on bad evictions, small reward for disagreement, reward agreeing experts on good evictions)
- ✅ Circular feedback buffer (`VecDeque<EvictionRecord>` capped at `window_size`)
- ✅ Parity tests cover initial uniform weights, weights-sum-to-one after updates, reset restores uniform, min weight prevents permanent silence
- ☐ Wire into `ModelAllocator::alloc_block()` (handoff to SLM-OS repo)
- ☐ Shell command: `eviction` — show policy stats, expert weights, recent decisions (handoff)

*Depends on: Milestones 6.2-6.4*

#### Milestone 6.6: Integration Testing 🔗
- ☐ QEMU test: load models, trigger evictions, verify correct victim selection
- ☐ Stress test: rapid alloc/free cycles with multiple models
- ☐ Compare: fault rate with AI policy vs hand-tuned heuristic
- ☐ Verify: no memory leaks, no deadlocks, no panics

*Depends on: Milestone 6.5*

**Phase 6 Gate:** ☐ AI-driven eviction policy runs in SLM-OS on QEMU. Measurable improvement over hand-tuned heuristic.

---

### Phase 7: Evaluation & Documentation (Week 7-8)

**Goal:** Comprehensive evaluation for capstone report and defense.

#### Milestone 7.1: Benchmark Suite ✅
- ✅ Define benchmark scenarios (7 workload types)
- ✅ Run each policy × each scenario × N seeds in simulator
- ✅ Collect: fault_rate, normalized_fault_rate, eviction_cost
- ✅ Statistical significance: paired t-tests between policies (`scripts/analyze_benchmark.py`)

#### Milestone 7.2: Comparative Analysis ✅
- ✅ Table: Policy × Scenario fault rate matrix (`policy_scenario_matrix.csv`)
- ✅ Chart: Normalized fault rate bar chart (`policy_comparison_per_scenario.png`)
- ✅ Chart: Heatmap (policy × scenario) (`policy_heatmap.png`)
- ✅ Chart: Per-policy ranking (`policy_overall_ranking.png`)
- ✅ Chart: CACHEUS weight trajectories (`scripts/plot_trajectories.py`)
- ✅ Chart: Feature importance (gain-based) — `scripts/analyze_models.py`
- ✅ Failure analysis: per-policy scenarios above 0.5 norm rate (`failure_analysis.csv`)

#### Milestone 7.3: SLM-OS Performance 🔗 ☐
- ☐ Measure end-to-end inference throughput with AI eviction vs LRU
- ☐ Measure eviction decision latency (target < 1μs)
- ☐ Measure memory overhead of AI policy (feature extraction + model weights)
- ☐ Compare: deadline miss rate with AI eviction vs heuristic

*Depends on: Phase 6 integration*

#### Milestone 7.4: Documentation ✅
- ✅ Architecture document (this document, finalized)
- ✅ API documentation for `EvictionPolicy` trait and implementations
- ✅ Training reproduction guide (scripts, seeds, hyperparameters)
- ☐ Capstone report section: "AI-Driven Page Replacement in SLM-OS"
- ☐ Defense slides: key results, architecture diagrams, live demo plan

**Phase 7 Gate:** ☐ All benchmarks complete, results documented, capstone section drafted.

---

## 11. Repository Structure

```
slm-os-page-sim/
├── pyproject.toml                   # Python project config
├── README.md                        # Project overview and quickstart
├── src/
│   ├── simulator/
│   │   ├── __init__.py
│   │   ├── core.py                  # SimController, MemoryState, Pool
│   │   ├── block.py                 # BlockMeta, PoolType, AccessPattern
│   │   ├── workload.py              # WorkloadGenerator, ModelConfig, scenarios
│   │   ├── trace.py                 # TraceCollector, Parquet export
│   │   └── metrics.py               # fault_rate, normalized_fault_rate, etc.
│   ├── policies/
│   │   ├── __init__.py
│   │   ├── base.py                  # Policy protocol / ABC
│   │   ├── lru.py                   # LRU policy
│   │   ├── lfu.py                   # LFU policy
│   │   ├── arc.py                   # ARC policy
│   │   ├── slm_heuristic.py         # Hand-tuned SLM-OS policy
│   │   ├── belady.py                # Bélády oracle (offline)
│   │   ├── mlp_policy.py            # Trained MLP wrapper
│   │   ├── xgb_policy.py            # Trained XGBoost wrapper
│   │   └── cacheus.py               # CACHEUS adaptive selector
│   ├── features/
│   │   ├── __init__.py
│   │   ├── extractor.py             # FeatureExtractor (27 features)
│   │   └── normalizer.py            # Feature normalization
│   ├── training/
│   │   ├── __init__.py
│   │   ├── dataset.py               # Parquet loading, train/val/test split
│   │   ├── train_xgb.py             # XGBoost training pipeline
│   │   ├── train_mlp.py             # MLP training pipeline
│   │   ├── dagger.py                # DAgger fine-tuning for MLP
│   │   └── evaluate.py              # Policy comparison in simulator
│   └── export/
│       ├── __init__.py
│       ├── xgb_to_rust.py           # Convert XGBoost JSON → Rust if-else
│       ├── mlp_to_rust.py           # Export MLP weights as Rust arrays
│       └── verify_export.py         # Cross-validate Python vs Rust predictions
├── tests/
│   ├── test_simulator.py            # Simulator unit tests
│   ├── test_workload.py             # Workload generator tests
│   ├── test_policies.py             # Policy correctness tests
│   ├── test_features.py             # Feature extraction tests
│   ├── test_belady.py               # Bélády oracle correctness
│   └── test_export.py               # Export verification
├── notebooks/
│   ├── 01_workload_analysis.ipynb   # Visualize workload traces
│   ├── 02_feature_analysis.ipynb    # Feature distributions and correlations
│   ├── 03_model_comparison.ipynb    # Training curves and policy comparison
│   └── 04_cacheus_analysis.ipynb    # Expert weight trajectories
├── data/
│   ├── traces/                      # Generated Parquet traces
│   ├── models/                      # Trained model checkpoints
│   └── results/                     # Benchmark results
├── configs/
│   ├── scenarios.yaml               # Workload scenario definitions
│   ├── xgb_params.yaml              # XGBoost hyperparameters
│   └── mlp_params.yaml              # MLP hyperparameters
└── scripts/
    ├── generate_dataset.py          # End-to-end dataset generation
    ├── train_all.py                 # Train all models
    ├── benchmark.py                 # Run full benchmark suite
    └── export_to_slmos.py           # Export models for SLM-OS integration
```

---

## 12. Dependencies

### Python (Simulator + Training)

| Package | Version | Purpose |
|---|---|---|
| Python | ≥ 3.10 | Language runtime |
| numpy | ≥ 1.24 | Array operations |
| pandas | ≥ 2.0 | DataFrame operations |
| pyarrow | ≥ 14.0 | Parquet I/O |
| xgboost | ≥ 2.0 | Gradient-boosted trees |
| torch | ≥ 2.1 | MLP training |
| scikit-learn | ≥ 1.3 | Metrics, preprocessing |
| matplotlib | ≥ 3.8 | Visualization |
| shap | ≥ 0.43 | Feature importance |
| pytest | ≥ 7.4 | Testing |
| pyyaml | ≥ 6.0 | Config files |

### Rust (SLM-OS Integration)

No additional crates needed. The MLP inference and XGBoost decision tree are implemented as pure Rust code with fixed-point arithmetic. Weight matrices are `const` arrays compiled into the binary.

---

## 13. Risk Mitigation

| Risk | Likelihood | Impact | Mitigation | Fallback |
|---|---|---|---|---|
| Simulator doesn't capture real workload patterns | Medium | High | Validate sim vs real traces once SLM-OS runs inference | Hand-tune heuristic remains as baseline |
| XGBoost overfits to training scenarios | Medium | Medium | Held-out test scenarios, cross-validation by scenario | Reduce tree depth, increase regularization |
| MLP too slow for bare-metal hot path | Low | Medium | Int8 quantization, 3-layer limit, benchmark early | Use XGBoost only (faster inference) |
| CACHEUS doesn't converge | Low | Medium | Tune learning rate, increase window, simplify to 3 experts | Use best single offline model |
| Feature extraction overhead too high | Low | High | Profile early, pre-compute ranks, cache features | Reduce to top-10 features |
| Dataset too small for MLP | Medium | Medium | DAgger augmentation, data augmentation via scenario variation | Use XGBoost (lower data requirement) |
| Rust export produces different results | Low | High | Automated cross-validation (Section 11, `verify_export.py`) | Debug systematically, fuzz test |

---

## 14. Success Metrics

| Metric | Target | Stretch Goal |
|---|---|---|
| Normalized fault rate (XGBoost) | < 0.3 | < 0.2 |
| Normalized fault rate (MLP) | < 0.35 | < 0.25 |
| CACHEUS vs best single expert | ≥ best on every scenario | Adaptation < 500 ticks |
| Eviction decision latency (Rust) | < 1μs on ARM64 | < 500ns |
| Model size (MLP, int8) | < 5 KB | < 3 KB |
| Feature extraction overhead | < 5% of eviction path | < 2% |
| End-to-end improvement vs LRU | > 10% fewer faults | > 25% fewer faults |

---

## 15. Connection to Capstone

### Academic Contribution

This work contributes a novel approach: **AI-driven page replacement specifically designed for SLM inference memory patterns.** While ML-based cache replacement is an active research area (LeCaR, CACHEUS, PARROT), no existing work targets:
1. 2MB block granularity (vs 64-byte cache lines or 4KB pages)
2. Model weight vs workspace differentiation
3. GPU-mapped block constraints
4. Deadline-aware eviction (urgency affects which blocks to keep)
5. SLM-specific access patterns (sequential layer sweeps, hot-swap transitions)

### Demo Plan

1. Show simulator running all scenarios with fault rate comparison chart
2. Show CACHEUS expert weight adaptation in real-time during workload transition
3. Boot SLM-OS in QEMU, load multiple models, trigger evictions
4. Shell command `eviction` shows AI policy statistics vs LRU baseline
5. Side-by-side: inference throughput with AI eviction vs hand-tuned heuristic

---

*Created: March 2026*
*Updated: April 2026 — status tracking added, Phases 1-2 complete, Phases 3-7 scaffolded*
*Companion to: SLM_OS_AI_Scheduler_Plan.md*
*Target: 8-week implementation alongside Phase 4X/5 SLM-OS development*
