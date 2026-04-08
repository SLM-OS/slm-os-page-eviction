# Simulator Reference

The discrete-event simulator models SLM-OS's dual-pool model memory system. All simulation code lives in `src/simulator/`.

## Core Components

### BlockMeta (`block.py`)

Represents a single 2MB memory block with full metadata:

```python
@dataclass
class BlockMeta:
    block_id: int           # Globally unique across both pools
    pool_type: PoolType     # WEIGHT or WORKSPACE
    model_id: int           # Owning model (0 = unassigned)
    layer_idx: int          # Layer position (-1 for workspace)
    state: BlockState       # FREE or ALLOCATED
    ref_count: int          # Active task references
    gpu_mapped: bool        # Currently GPU-accessible
    last_access_time: int   # Tick of most recent access
    access_count: int       # Lifetime access count
    load_time: int          # Tick when block was loaded
    access_pattern: AccessPattern  # SEQUENTIAL/RANDOM/STRIDED/BURST
    is_dirty: bool          # Modified since load
```

A block is **evictable** when: `state == ALLOCATED and ref_count == 0 and not gpu_mapped`.

### Pool (`core.py`)

A fixed-size array of `BlockMeta`. Provides:

- `find_free_block()` -- first block with `state == FREE`
- `get_evictable_blocks()` -- all blocks passing `is_evictable()`
- `recency_rank(block)` -- rank by `last_access_time` (0 = most recent)
- `frequency_rank(block)` -- rank by `access_count` (0 = most accessed)

### MemoryState (`core.py`)

Wraps `weight_pool` and `workspace_pool`. Assigns unique block IDs across both pools (workspace IDs start at `weight_blocks` offset). Provides `get_pool(pool_type)` and `get_block_by_id(id)`.

### SimController (`core.py`)

Drives the simulation loop. Key state:

- `tick` -- current simulation time
- `_loaded_models` -- maps model_id to set of block_ids
- `_model_priorities` -- maps model_id to priority level
- `_active_inferences` -- maps model_id to active inference count
- `_recent_faults / _recent_accesses` -- sliding window for fault rate

`process_access(request)` handles one access. `run(requests)` processes a full sequence.

`get_global_state()` produces a `GlobalState` snapshot for policy decisions, computing pool utilizations, loaded model count, and recent fault rate.

### GlobalState (`core.py`)

Immutable snapshot of simulator state at an eviction decision point. Contains 12 fields matching the global features in the feature vector (Section 2.2 of the plan).

## Workload Generator (`workload.py`)

Produces `AccessRequest` sequences for 7 scenarios:

| Scenario | Description | Primary stress |
|----------|-------------|----------------|
| `single_inference` | One model, repeated inference sweeps | Sequential weight access |
| `multi_model` | 2-4 models with interleaved inference | Pool contention |
| `hot_swap` | Model replacement mid-run | Cold/hot transition |
| `burst_load` | Rapid concurrent model loads | Spike pressure |
| `mixed_priority` | Critical + normal priority models | Priority-aware eviction |
| `gpu_contention` | GPU-mapped blocks competing | GPU constraints |
| `adversarial` | Deliberately pathological patterns | Worst-case stress |

Each scenario accepts a random seed for reproducibility. Model archetypes (tiny, small, medium, large, attn, critical) define layer counts, workspace needs, and priority levels.

Usage:
```python
wg = WorkloadGenerator(seed=42)
requests = wg.generate_scenario("hot_swap")
```

## Trace Collection (`trace.py`)

`TraceCollector` records two event types during simulation:

- **AccessEvent** -- tick, block_id, model_id, layer_idx, pool_type, is_hit
- **EvictionEvent** -- tick, eviction_id, victim info, candidate_block_ids, global_state

Export to Parquet via `export_accesses_parquet()` / `export_evictions_parquet()` or to DataFrames via `to_access_dataframe()` / `to_eviction_dataframe()`.

## Metrics (`metrics.py`)

- `PolicyMetrics` -- dataclass collecting total_accesses, total_faults, total_evictions, fault_rate
- `compute_fault_rate(faults, accesses)` -- faults / accesses
- `compute_normalized_fault_rate(policy_faults, optimal_faults, lru_faults)` -- 0.0 = Belady optimal, 1.0 = LRU baseline
