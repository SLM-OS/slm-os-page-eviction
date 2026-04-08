# SLM-OS AI-Driven Page Replacement Simulator

Discrete-event simulator and ML training pipeline for learning optimal 2MB block eviction decisions in the SLM-OS model memory allocator. Part of the [CS-496 Capstone SLM Operating System](https://github.com/CSejersen/CS-496-Capstone-SLM-Operating-System) project.

## What This Does

SLM-OS manages model weights and inference workspace in two memory pools of 2MB blocks. When a pool is full and a new block needs to be loaded, the system must choose which resident block to evict. This project:

1. **Simulates** the dual-pool memory system with 7 synthetic workload scenarios matching SLM inference patterns (sequential weight sweeps, hot-swap, burst load, GPU contention, etc.)
2. **Generates training data** by running a Belady optimal oracle (provably best eviction decisions) and extracting 27-feature vectors at each eviction point
3. **Trains ML models** (XGBoost + MLP) to predict the optimal eviction target from the feature vector
4. **Evaluates** against classical baselines (LRU, LFU, ARC) and a hand-tuned SLM-OS heuristic
5. **Exports** trained models as standalone Rust code (if-else chains for XGBoost, int8 const arrays for MLP) for bare-metal deployment in SLM-OS

## Architecture

```
WorkloadGenerator → SimController → TraceCollector → Belady Labels → FeatureExtractor
                         │                                                    │
                    EvictionPolicy                                    27-feature vectors
                    (LRU/LFU/ARC/                                          │
                     SLM/Belady/                                 ┌─────────┴─────────┐
                     XGB/MLP/                                    │                   │
                     CACHEUS)                               train_xgb.py        train_mlp.py
                                                                 │                   │
                                                            XGB Booster    PageReplacementMLP
                                                                 │                   │
                                                          xgb_to_rust.py     mlp_to_rust.py
                                                                 │                   │
                                                           Rust if-else      Rust int8 arrays
                                                                 └─────────┬─────────┘
                                                                           │
                                                                    SLM-OS runtime
                                                              (runtime/src/mm/model_mem.rs)
```

## Quick Start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
python -m pytest tests/    # 61 tests, all passing
```

Generate a small dataset and verify the pipeline:

```bash
python scripts/generate_dataset.py --seeds 1 --weight-blocks 16 --workspace-blocks 8
```

See [docs/getting-started.md](docs/getting-started.md) for full setup and usage instructions.

## Implementation Status

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | Simulator core, workloads, classical policies, trace collection | ✅ Complete |
| Phase 2 | Feature extraction (27-dim), normalization, Belady labeling, dataset assembly | ✅ Complete |
| Phase 3 | XGBoost training pipeline, cross-validation | Training code complete, tuning/eval pending |
| Phase 4 | MLP training, DAgger fine-tuning, quantization | Training code complete, eval pending |
| Phase 5 | CACHEUS adaptive expert selector | Selector implemented, online eval pending |
| Phase 6 | Rust export pipeline (XGBoost if-else, MLP int8) | Export code complete, SLM-OS integration pending |
| Phase 7 | Benchmark suite, comparative analysis, documentation | Benchmark framework complete, full analysis pending |

61 unit tests cover the simulator, workload generator, all policies, feature extraction, Belady oracle, and export verification.

See [SLM_OS_AI_Page_Replacement_Plan.md](SLM_OS_AI_Page_Replacement_Plan.md) for the full plan with per-task status tracking.

## Eviction Policies

| Policy | Type | Description |
|--------|------|-------------|
| LRU | Classical | Evict least recently used |
| LFU | Classical | Evict least frequently used |
| ARC | Classical | Adaptive recency/frequency balance with ghost lists |
| SLM-Heuristic | Hand-tuned | Workspace-before-weights priority, inactive model preference |
| Belady | Oracle | Offline optimal (evict block with farthest next access) |
| XGBoost | ML | Depth-6 tree ensemble, sub-microsecond inference |
| MLP | ML | 3-layer network (64-32-16), dual-head classification + reuse regression |
| CACHEUS | Ensemble | Weighted expert selector with online regret-based adaptation |

## Feature Vector

Each eviction candidate is described by **27 features** (or 26 without the optional `predicted_reuse_dist`):

- **15 per-block**: recency rank, frequency rank, access count, time since access/load, ref count, GPU mapped, pool type, dirty, layer position, model priority, active inferences, access pattern, predicted reuse distance, eviction cost
- **12 global**: pool utilizations, loaded model count, GPU mapped total, pending loads, average priority, deadline pressure, recent fault rate, hot-swap flag, requesting block context

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Block size | 2MB | Matches SLM-OS ModelAllocator |
| Primary model | XGBoost (depth-6) | Sub-microsecond inference, exports to Rust if-else |
| Secondary model | MLP (64-32-16) | Differentiable, serves as CACHEUS expert |
| Training labels | Belady optimal | Provably correct upper bound |
| Online adaptation | CACHEUS ensemble | Adapts to unknown workload distributions at runtime |
| Dual feature mode | 27 vs 26 features | Empirically compare heuristic reuse estimate vs implicit learning |

## Documentation

| Document | Description |
|----------|-------------|
| [SLM_OS_AI_Page_Replacement_Plan.md](SLM_OS_AI_Page_Replacement_Plan.md) | Full architecture and implementation plan with task status |
| [docs/architecture.md](docs/architecture.md) | System architecture and data flow |
| [docs/simulator.md](docs/simulator.md) | Simulator components and event loop |
| [docs/policies.md](docs/policies.md) | Eviction policy implementations |
| [docs/features.md](docs/features.md) | Feature engineering and normalization |
| [docs/training.md](docs/training.md) | Training pipeline, dataset generation, DAgger |
| [docs/export.md](docs/export.md) | Rust export pipeline and verification |
| [docs/getting-started.md](docs/getting-started.md) | Setup, usage, and project layout |

## Dependencies

**Core** (simulator, features, classical policies): numpy, pandas, pyarrow, scikit-learn, pyyaml, pytest

**ML** (training, ML policies, export): xgboost, torch

**Target** (SLM-OS integration): Pure Rust, no runtime dependencies. Models compile to ~50KB (XGBoost) and ~4KB (MLP int8).

## Repository Structure

```
slm-os-page-sim/
├── src/
│   ├── simulator/     # Core simulator: block, pool, controller, workload, trace, metrics
│   ├── policies/      # 8 eviction policies: LRU, LFU, ARC, SLM, Belady, XGB, MLP, CACHEUS
│   ├── features/      # 27-feature extraction and normalization
│   ├── training/      # Dataset, XGBoost/MLP training, DAgger, evaluation
│   └── export/        # Rust code generation and cross-validation
├── tests/             # 61 unit tests
├── scripts/           # CLI: generate_dataset, train_all, benchmark, export_to_slmos
├── configs/           # YAML: scenarios, xgb_params, mlp_params
├── docs/              # Detailed documentation
└── data/              # Generated artifacts (gitignored)
```
