#!/usr/bin/env python3
"""End-to-end dataset generation pipeline.

Runs the simulator across all 7 scenarios x 5 seeds, collects traces
with Belady-optimal labels, extracts features, and exports to Parquet.

Usage:
    python scripts/generate_dataset.py [--output-dir data/traces] [--seeds 5]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.extractor import FeatureConfig, FeatureExtractor
from src.features.normalizer import FeatureNormalizer
from src.policies.base import EvictionPolicy
from src.policies.belady import BeladyOracle
from src.policies.lru import LRUPolicy
from src.simulator.block import BlockMeta
from src.simulator.core import GlobalState, MemoryState, Pool, SimController
from src.simulator.trace import TraceCollector
from src.simulator.workload import WorkloadGenerator


class _LabelingPolicy(EvictionPolicy):
    """Wraps Belady oracle, capturing features + labels at each eviction."""

    def __init__(
        self,
        oracle: BeladyOracle,
        extractor: FeatureExtractor,
        scenario: str,
        eviction_id_offset: int = 0,
    ):
        self.oracle = oracle
        self.extractor = extractor
        self.scenario = scenario
        self.rows: list[dict] = []
        self._eviction_id = eviction_id_offset

    def select_victim(
        self,
        candidates: list[BlockMeta],
        pool: Pool,
        global_state: GlobalState,
    ) -> int:
        optimal_idx, reuse_distances = self.oracle.label_eviction(
            candidates, pool, global_state
        )
        features = self.extractor.extract_candidate_features(
            candidates, pool, global_state
        )
        feature_names = self.extractor.config.feature_names

        for i in range(len(candidates)):
            row = {name: float(features[i, j]) for j, name in enumerate(feature_names)}
            row["is_optimal"] = 1 if i == optimal_idx else 0
            row["reuse_distance"] = reuse_distances[i]
            row["eviction_id"] = self._eviction_id
            row["scenario"] = self.scenario
            self.rows.append(row)

        self._eviction_id += 1
        return optimal_idx

    def name(self) -> str:
        return "Belady-Labeling"


def generate_dataset(
    output_dir: str = "data/traces",
    num_seeds: int = 5,
    use_predicted_reuse: bool = True,
    weight_blocks: int = 64,
    workspace_blocks: int = 32,
) -> None:
    """Generate the complete labeled dataset."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    config = FeatureConfig(use_predicted_reuse=use_predicted_reuse)
    extractor = FeatureExtractor(config)

    seeds = [42, 123, 456, 789, 1337][:num_seeds]
    scenarios = WorkloadGenerator.all_scenario_names()
    all_rows: list[dict] = []
    eviction_id_offset = 0

    for scenario in scenarios:
        for seed in seeds:
            print(f"Generating: {scenario} (seed={seed})")
            wg = WorkloadGenerator(seed=seed)
            requests = wg.generate_scenario(scenario)

            if not requests:
                continue

            # Run with LRU to generate an access trace for Belady's oracle
            memory = MemoryState(weight_blocks, workspace_blocks)
            trace = TraceCollector(scenario=scenario, seed=seed)
            lru = LRUPolicy()
            sim = SimController(memory=memory, policy=lru, trace_collector=trace)
            sim.run(requests)

            # Create Belady oracle from the complete access trace
            oracle = BeladyOracle(future_accesses=trace.access_events)

            # Re-run with labeling policy that captures features in-situ
            labeler = _LabelingPolicy(
                oracle, extractor, scenario, eviction_id_offset
            )
            memory2 = MemoryState(weight_blocks, workspace_blocks)
            trace2 = TraceCollector(scenario=scenario, seed=seed)
            sim2 = SimController(
                memory=memory2, policy=labeler, trace_collector=trace2
            )
            sim2.run(requests)

            all_rows.extend(labeler.rows)
            eviction_id_offset = labeler._eviction_id

            print(
                f"  {scenario}/{seed}: "
                f"accesses={sim2.total_accesses}, "
                f"faults={sim2.total_faults}, "
                f"evictions={sim2.total_evictions}, "
                f"rows={len(labeler.rows)}"
            )

    if all_rows:
        df = pd.DataFrame(all_rows)
        output_file = output_path / "eviction_events.parquet"
        df.to_parquet(output_file, index=False)
        print(f"\nDataset saved: {output_file}")
        print(f"Total rows: {len(df)}")
        print(f"Features: {config.num_features}")
        print(f"Eviction events: {eviction_id_offset}")
        print(f"Class balance (optimal): {df['is_optimal'].mean():.3f}")
    else:
        print("\nNo eviction events generated. Check pool sizes and workloads.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate training dataset")
    parser.add_argument(
        "--output-dir", default="data/traces",
        help="Output directory for Parquet files",
    )
    parser.add_argument(
        "--seeds", type=int, default=5,
        help="Number of random seeds per scenario",
    )
    parser.add_argument(
        "--no-predicted-reuse", action="store_true",
        help="Exclude predicted_reuse_dist feature",
    )
    parser.add_argument(
        "--weight-blocks", type=int, default=64,
        help="Number of weight pool blocks",
    )
    parser.add_argument(
        "--workspace-blocks", type=int, default=32,
        help="Number of workspace pool blocks",
    )
    args = parser.parse_args()

    generate_dataset(
        output_dir=args.output_dir,
        num_seeds=args.seeds,
        use_predicted_reuse=not args.no_predicted_reuse,
        weight_blocks=args.weight_blocks,
        workspace_blocks=args.workspace_blocks,
    )


if __name__ == "__main__":
    main()
