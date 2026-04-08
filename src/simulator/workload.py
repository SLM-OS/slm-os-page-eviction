"""Workload generator: model archetypes and 7 scenario types.

Generates sequences of AccessRequest events that model realistic SLM-OS
inference workloads. Each scenario exercises different memory pressure
patterns and access characteristics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from .block import PoolType, AccessPattern


class ModelConfig(NamedTuple):
    """SLM model archetype (Section 1 of the plan)."""
    model_id: int
    name: str
    num_layers: int
    workspace_blocks: int   # Workspace blocks needed per inference
    weight_blocks: int      # Total weight blocks (layers * blocks_per_layer approx)
    priority: int           # Scheduling priority (0 = lowest, 7 = highest)
    uses_gpu: bool          # Requires GPU-mapped blocks
    access_order: str       # "sequential" or "attention"


# Pre-defined model archetypes matching the plan's Table 1
MODELS: dict[str, ModelConfig] = {
    "tiny":     ModelConfig(0, "tiny-4L",     4,  1,   50,  3, False, "sequential"),
    "small":    ModelConfig(1, "small-8L",    8,  2,  100,  3, False, "sequential"),
    "medium":   ModelConfig(2, "medium-16L", 16,  3,  200,  4, False, "sequential"),
    "large":    ModelConfig(3, "large-32L",  32,  4,  400,  5, True,  "sequential"),
    "attn":     ModelConfig(4, "attn-16L",   16,  6,  250,  4, True,  "attention"),
    "critical": ModelConfig(5, "crit-8L",     8,  2,  100,  7, True,  "sequential"),
}


@dataclass
class AccessRequest:
    """A single memory access request in the simulation trace."""
    tick: int
    model_id: int
    layer_idx: int              # -1 for workspace blocks
    pool_type: PoolType
    access_pattern: AccessPattern
    priority: int = 3
    deadline_pressure: float = 0.0  # [0, 1] urgency
    is_gpu: bool = False


class WorkloadGenerator:
    """Generates access request sequences for the 7 workload scenarios.

    Each scenario produces a list of AccessRequest events ordered by tick.
    A random seed ensures reproducibility across runs.
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.seed = seed

    def gen_single_inference(
        self,
        model: ModelConfig | None = None,
        num_inferences: int = 10,
        ticks_per_layer: int = 5,
    ) -> list[AccessRequest]:
        """Scenario 1: Single model steady-state inference.

        Sequential layer sweep repeated for multiple inference passes.
        Workspace allocated at start of each inference, freed at end.
        """
        if model is None:
            model = MODELS["medium"]

        requests: list[AccessRequest] = []
        tick = 0

        for _ in range(num_inferences):
            # Allocate workspace at inference start
            for ws in range(model.workspace_blocks):
                requests.append(AccessRequest(
                    tick=tick,
                    model_id=model.model_id,
                    layer_idx=-1,
                    pool_type=PoolType.WORKSPACE,
                    access_pattern=AccessPattern.BURST,
                    priority=model.priority,
                    is_gpu=model.uses_gpu,
                ))
                tick += 1

            # Sequential weight access through layers
            for layer in range(model.num_layers):
                # Each layer may use multiple weight blocks
                num_weight_blocks = model.weight_blocks // model.num_layers
                for wb in range(num_weight_blocks):
                    requests.append(AccessRequest(
                        tick=tick,
                        model_id=model.model_id,
                        layer_idx=layer * num_weight_blocks + wb,
                        pool_type=PoolType.WEIGHT,
                        access_pattern=AccessPattern.SEQUENTIAL,
                        priority=model.priority,
                        is_gpu=model.uses_gpu,
                    ))
                    tick += 1

                # Workspace access during layer computation
                for ws in range(model.workspace_blocks):
                    requests.append(AccessRequest(
                        tick=tick,
                        model_id=model.model_id,
                        layer_idx=-1,
                        pool_type=PoolType.WORKSPACE,
                        access_pattern=AccessPattern.BURST,
                        priority=model.priority,
                        is_gpu=model.uses_gpu,
                    ))
                tick += ticks_per_layer

        return requests

    def gen_multi_model(
        self,
        models: list[ModelConfig] | None = None,
        num_inferences_each: int = 5,
        interleave_ticks: int = 3,
    ) -> list[AccessRequest]:
        """Scenario 2: Multiple models with interleaved concurrent inference.

        Models run concurrently, competing for pool space. Exercises the
        policy's ability to balance recency vs. model importance.
        """
        if models is None:
            models = [MODELS["tiny"], MODELS["small"], MODELS["medium"]]

        all_requests: list[AccessRequest] = []

        for model in models:
            requests = self.gen_single_inference(
                model, num_inferences=num_inferences_each
            )
            # Offset ticks to interleave models
            offset = self.rng.integers(0, interleave_ticks * 10)
            for r in requests:
                r.tick += offset
            all_requests.extend(requests)

        # Sort by tick to produce a single merged timeline
        all_requests.sort(key=lambda r: r.tick)
        return all_requests

    def gen_hot_swap(
        self,
        model_old: ModelConfig | None = None,
        model_new: ModelConfig | None = None,
        swap_tick: int = 5000,
        warmup_inferences: int = 5,
        post_swap_inferences: int = 5,
    ) -> list[AccessRequest]:
        """Scenario 3: Model hot-swap mid-run.

        An old model runs for a warmup phase, then a new model is loaded,
        replacing the old model's blocks. Tests the policy's ability to
        quickly evict stale blocks from the departing model.
        """
        if model_old is None:
            model_old = MODELS["medium"]
        if model_new is None:
            model_new = MODELS["large"]

        requests: list[AccessRequest] = []

        # Phase 1: Old model running
        old_requests = self.gen_single_inference(
            model_old, num_inferences=warmup_inferences
        )
        requests.extend(old_requests)

        # Phase 2: New model loads (starting at swap_tick)
        new_requests = self.gen_single_inference(
            model_new, num_inferences=post_swap_inferences
        )
        for r in new_requests:
            r.tick += swap_tick
        requests.extend(new_requests)

        requests.sort(key=lambda r: r.tick)
        return requests

    def gen_burst_load(
        self,
        model: ModelConfig | None = None,
        burst_size: int = 20,
        burst_interval: int = 100,
        num_bursts: int = 10,
        baseline_inferences: int = 3,
    ) -> list[AccessRequest]:
        """Scenario 4: Memory pressure spike (burst load).

        Baseline steady-state inference with periodic bursts of rapid
        allocations that overwhelm the pool and force rapid evictions.
        """
        if model is None:
            model = MODELS["small"]

        requests: list[AccessRequest] = []

        # Baseline inference
        baseline = self.gen_single_inference(
            model, num_inferences=baseline_inferences
        )
        requests.extend(baseline)

        # Periodic bursts
        tick = burst_interval
        burst_model = MODELS["tiny"]
        for _ in range(num_bursts):
            for i in range(burst_size):
                requests.append(AccessRequest(
                    tick=tick + i,
                    model_id=burst_model.model_id,
                    layer_idx=i % burst_model.num_layers,
                    pool_type=PoolType.WEIGHT,
                    access_pattern=AccessPattern.BURST,
                    priority=burst_model.priority,
                ))
            tick += burst_interval

        requests.sort(key=lambda r: r.tick)
        return requests

    def gen_mixed_priority(
        self,
        num_inferences: int = 5,
    ) -> list[AccessRequest]:
        """Scenario 5: Priority-aware workload.

        High-priority (critical) and low-priority models run concurrently.
        Tests whether the policy respects priority when choosing victims.
        """
        high = MODELS["critical"]
        low = MODELS["tiny"]

        high_requests = self.gen_single_inference(
            high, num_inferences=num_inferences
        )
        low_requests = self.gen_single_inference(
            low, num_inferences=num_inferences
        )

        # Add deadline pressure to high-priority requests
        for r in high_requests:
            r.deadline_pressure = 0.8

        # Interleave
        all_requests = high_requests + low_requests
        all_requests.sort(key=lambda r: r.tick)
        return all_requests

    def gen_gpu_contention(
        self,
        num_inferences: int = 5,
    ) -> list[AccessRequest]:
        """Scenario 6: GPU-mapped block constraints.

        Multiple GPU-using models compete, creating blocks that cannot be
        evicted until GPU-unmapped. Tests the policy's handling of the
        gpu_mapped constraint.
        """
        models = [MODELS["large"], MODELS["attn"]]
        all_requests: list[AccessRequest] = []

        for model in models:
            requests = self.gen_single_inference(
                model, num_inferences=num_inferences
            )
            # Mark GPU requests
            for r in requests:
                r.is_gpu = True
            offset = self.rng.integers(0, 50)
            for r in requests:
                r.tick += offset
            all_requests.extend(requests)

        all_requests.sort(key=lambda r: r.tick)
        return all_requests

    def gen_adversarial(
        self,
        pool_size: int = 32,
        num_accesses: int = 5000,
    ) -> list[AccessRequest]:
        """Scenario 7: Adversarial / pathological access patterns.

        Designed to stress-test policies with patterns that defeat simple
        heuristics: cyclic scans slightly larger than the pool, random
        jumps, and LRU-hostile sequences.
        """
        requests: list[AccessRequest] = []
        # Cyclic scan over pool_size + 1 blocks (defeats LRU)
        cycle_len = pool_size + 1
        model = MODELS["small"]

        for tick in range(num_accesses):
            layer_idx = tick % cycle_len
            requests.append(AccessRequest(
                tick=tick,
                model_id=model.model_id,
                layer_idx=layer_idx,
                pool_type=PoolType.WEIGHT,
                access_pattern=AccessPattern.SEQUENTIAL,
                priority=model.priority,
            ))

        return requests

    def generate_scenario(
        self,
        scenario_name: str,
        **kwargs,
    ) -> list[AccessRequest]:
        """Generate a named scenario. Dispatches to the appropriate method."""
        generators = {
            "single_inference": self.gen_single_inference,
            "multi_model": self.gen_multi_model,
            "hot_swap": self.gen_hot_swap,
            "burst_load": self.gen_burst_load,
            "mixed_priority": self.gen_mixed_priority,
            "gpu_contention": self.gen_gpu_contention,
            "adversarial": self.gen_adversarial,
        }
        if scenario_name not in generators:
            raise ValueError(
                f"Unknown scenario: {scenario_name}. "
                f"Available: {list(generators.keys())}"
            )
        return generators[scenario_name](**kwargs)

    @staticmethod
    def all_scenario_names() -> list[str]:
        """Return all available scenario names."""
        return [
            "single_inference",
            "multi_model",
            "hot_swap",
            "burst_load",
            "mixed_priority",
            "gpu_contention",
            "adversarial",
        ]
