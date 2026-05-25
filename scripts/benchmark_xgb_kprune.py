#!/usr/bin/env python3
"""K-pruned XGBoost hit-rate sweep for #961.

Loads the trained 200-tree XGBoost model, evaluates it on the full
scenario × seed grid at several prefix-tree-counts K, and reports
normalized fault rate (LRU=1.0, Belady=0.0) per K per scenario.

Goal: decide whether tree-pruning (use first K trees only) preserves
workload hit-rate well enough to ship as the runtime model, so we can
hit the <1 µs hot-path target without sacrificing eviction quality.

Usage:
    .venv/bin/python scripts/benchmark_xgb_kprune.py [--seeds 3] [--ks 8,16,32,64,200]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from src.features.extractor import FeatureConfig, FeatureExtractor
from src.features.normalizer import FeatureNormalizer
from src.policies.base import EvictionPolicy
from src.policies.belady import BeladyOracle
from src.policies.lru import LRUPolicy
from src.simulator.core import MemoryState, SimController
from src.simulator.metrics import compute_normalized_fault_rate
from src.simulator.trace import TraceCollector
from src.simulator.workload import WorkloadGenerator
from src.training.evaluate import PolicyEvaluator
from src.training.train_xgb import load_model as load_xgb


class XGBPolicyTopK(EvictionPolicy):
    """XGBoost policy that only uses the first K trees in the ensemble.

    Mirrors XGBPolicy but slices the predict call with iteration_range=(0, K).
    K=0 or K>=len(trees) uses the full model.
    """

    def __init__(
        self,
        model: xgb.Booster,
        n_trees: int,
        feature_config: FeatureConfig | None = None,
    ):
        self._model = model
        self._n_trees = n_trees
        self._config = feature_config or FeatureConfig()
        self._extractor = FeatureExtractor(self._config)
        self._normalizer = FeatureNormalizer(self._config)
        self._total_trees = len(model.get_dump())

    def select_victim(self, candidates, pool, global_state) -> int:
        scores = self.score(candidates, pool, global_state)
        return int(np.argmax(scores))

    def score(self, candidates, pool, global_state):
        if not candidates:
            return []
        features = self._extractor.extract_candidate_features(
            candidates, pool, global_state
        )
        features = self._normalizer.normalize(
            features, pool_size=pool.num_blocks
        )
        dmatrix = xgb.DMatrix(
            features, feature_names=self._config.feature_names
        )
        k = self._n_trees
        if k <= 0 or k >= self._total_trees:
            probs = self._model.predict(dmatrix)
        else:
            probs = self._model.predict(dmatrix, iteration_range=(0, k))
        return probs.tolist()

    def name(self) -> str:
        return f"XGB-K{self._n_trees}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", default="data/models/xgb_model.json",
        help="Path to trained XGBoost model JSON",
    )
    parser.add_argument(
        "--output-dir", default="data/results/kprune",
        help="Output directory for results CSVs",
    )
    parser.add_argument(
        "--seeds", type=int, default=3,
        help="Number of seeds (default 3 for speed; bump to 5 for parity with benchmark.py)",
    )
    parser.add_argument(
        "--ks", default="8,16,32,64,200",
        help="Comma-separated K values to evaluate",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_config = FeatureConfig()
    booster = load_xgb(args.model)
    total_trees = len(booster.get_dump())
    print(f"Loaded model: {total_trees} trees, "
          f"best_iteration={booster.attributes().get('best_iteration')}")

    ks = [int(k) for k in args.ks.split(",")]
    print(f"Evaluating K values: {ks}")

    seeds = [42, 123, 456, 789, 1337][:args.seeds]
    evaluator = PolicyEvaluator(weight_blocks=64, workspace_blocks=32)
    scenarios = WorkloadGenerator.all_scenario_names()

    policies = [XGBPolicyTopK(booster, k, feature_config) for k in ks]
    total_runs = len(scenarios) * len(seeds) * (len(policies) + 2)
    print(f"\nRunning {total_runs} simulations "
          f"({len(scenarios)} scenarios × {len(seeds)} seeds × "
          f"({len(policies)} XGB-K + LRU + Belady)) ...\n")

    all_results = []
    run_count = 0
    for scenario_name in scenarios:
        for seed in seeds:
            wg = WorkloadGenerator(seed=seed)
            requests = wg.generate_scenario(scenario_name)

            # LRU baseline for normalization
            lru = LRUPolicy()
            lru_res = evaluator.run_policy(lru, requests, scenario_name, seed)
            all_results.append(lru_res)
            run_count += 1
            print(f"  [{run_count}/{total_runs}] LRU / {scenario_name} / {seed}: "
                  f"faults={lru_res.metrics.total_faults}")

            # Belady oracle for normalization
            memory = MemoryState(evaluator.weight_blocks, evaluator.workspace_blocks)
            trace = TraceCollector(scenario=scenario_name, seed=seed)
            sim = SimController(
                memory=memory, policy=LRUPolicy(), trace_collector=trace
            )
            sim.run(requests)
            oracle = BeladyOracle(future_accesses=trace.access_events)
            oracle.reset()
            belady_res = evaluator.run_policy(
                oracle, requests, scenario_name, seed
            )
            all_results.append(belady_res)
            run_count += 1
            print(f"  [{run_count}/{total_runs}] Belady-Optimal / {scenario_name} / {seed}: "
                  f"faults={belady_res.metrics.total_faults}")

            for policy in policies:
                policy.reset()
                r = evaluator.run_policy(policy, requests, scenario_name, seed)
                all_results.append(r)
                run_count += 1
                print(f"  [{run_count}/{total_runs}] {policy.name()} / "
                      f"{scenario_name} / {seed}: faults={r.metrics.total_faults}")

    # Build long-form DataFrame
    index = {(r.policy_name, r.scenario, r.seed): r for r in all_results}
    rows = []
    for r in all_results:
        lru = index[("LRU", r.scenario, r.seed)].metrics.total_faults
        opt = index[("Belady-Optimal", r.scenario, r.seed)].metrics.total_faults
        norm = compute_normalized_fault_rate(r.metrics.total_faults, opt, lru)
        rows.append({
            "policy": r.policy_name,
            "scenario": r.scenario,
            "seed": r.seed,
            "total_accesses": r.metrics.total_accesses,
            "total_faults": r.metrics.total_faults,
            "fault_rate": r.metrics.fault_rate,
            "normalized_fault_rate": norm,
        })
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "kprune_raw.csv", index=False)

    # Pivot: rows = scenario, cols = K, values = mean normalized fault rate
    xgb_only = df[df.policy.str.startswith("XGB-K")].copy()
    xgb_only["k"] = xgb_only.policy.str.removeprefix("XGB-K").astype(int)
    pivot = xgb_only.pivot_table(
        index="scenario", columns="k",
        values="normalized_fault_rate", aggfunc="mean",
    )
    # Order columns by k ascending for readability
    pivot = pivot[sorted(pivot.columns)]

    # Reference column: baseline (largest K) normalized fault rate
    baseline_k = max(ks)
    pivot["delta_vs_K{0}".format(baseline_k)] = 0.0  # placeholder, computed below
    delta = pivot.subtract(pivot[baseline_k], axis=0)
    for k in ks:
        if k == baseline_k:
            continue
        pivot[f"Δ_K{k}"] = delta[k]
    pivot = pivot.drop(columns=[f"delta_vs_K{baseline_k}"])

    pivot.to_csv(output_dir / "kprune_pivot.csv")
    print("\n=== Mean normalized fault rate (lower = better; LRU=1.0, Belady=0.0) ===")
    print(pivot.to_string())

    # Aggregate across all scenarios+seeds
    overall = xgb_only.groupby("k").agg(
        mean_nfr=("normalized_fault_rate", "mean"),
        std_nfr=("normalized_fault_rate", "std"),
        median_nfr=("normalized_fault_rate", "median"),
    ).sort_index()
    baseline = overall.loc[baseline_k, "mean_nfr"]
    overall["delta_vs_baseline"] = overall["mean_nfr"] - baseline
    overall["pct_vs_baseline"] = (
        100.0 * (overall["mean_nfr"] - baseline) / max(abs(baseline), 1e-9)
    )
    overall.to_csv(output_dir / "kprune_overall.csv")
    print(f"\n=== Overall (across all scenarios+seeds), baseline = K={baseline_k} ===")
    print(overall.to_string())

    print(f"\nResults written to: {output_dir}/")


if __name__ == "__main__":
    main()
