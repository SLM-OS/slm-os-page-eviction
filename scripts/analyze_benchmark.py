#!/usr/bin/env python3
"""Phase 7 benchmark analysis: statistical tests, comparison tables, charts.

Reads benchmark_results.csv produced by scripts/benchmark.py and produces:
- Pivot tables of normalized fault rate per (policy, scenario)
- Paired t-tests between policy pairs (per scenario, across seeds)
- Bar charts: per-scenario comparison and per-policy overall
- Failure analysis: which scenarios each policy struggles on

Usage:
    python scripts/analyze_benchmark.py --input data/results/benchmark_results.csv \\
                                        --output-dir data/results/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def pivot_results(df: pd.DataFrame, value_col: str = "normalized_fault_rate") -> pd.DataFrame:
    """Build a (policy × scenario) pivot table of the mean of `value_col`."""
    summary = df.groupby(["policy", "scenario"])[value_col].mean().reset_index()
    pivot = summary.pivot(index="policy", columns="scenario", values=value_col)

    policy_order = [
        "Belady-Optimal", "XGBoost", "MLP", "CACHEUS",
        "ARC", "LRU", "LFU", "SLM-Heuristic",
    ]
    scenario_order = [
        "single_inference", "multi_model", "hot_swap", "burst_load",
        "mixed_priority", "gpu_contention", "adversarial",
    ]

    # Reindex but only for policies/scenarios actually present
    pivot = pivot.reindex([p for p in policy_order if p in pivot.index])
    pivot = pivot[[s for s in scenario_order if s in pivot.columns]]

    return pivot


def paired_ttest_pairs(
    df: pd.DataFrame,
    policy_a: str,
    policy_b: str,
    value_col: str = "normalized_fault_rate",
) -> dict:
    """Per-scenario paired t-test of policy_a vs policy_b across seeds.

    Returns dict keyed by scenario with t, p, dof, and significant flag (p<0.05).
    """
    results = {}
    for scenario in sorted(df["scenario"].unique()):
        a = df[(df["policy"] == policy_a) & (df["scenario"] == scenario)] \
            .sort_values("seed")[value_col].to_numpy()
        b = df[(df["policy"] == policy_b) & (df["scenario"] == scenario)] \
            .sort_values("seed")[value_col].to_numpy()

        if len(a) != len(b) or len(a) < 2:
            results[scenario] = {"valid": False, "reason": "insufficient seeds"}
            continue

        # Constant arrays (e.g. when both policies achieve identical results)
        # produce NaN p-values; treat as non-significant.
        if np.std(a - b) == 0:
            results[scenario] = {
                "valid": True,
                "t": 0.0,
                "p": 1.0,
                "dof": len(a) - 1,
                "significant": False,
                "mean_diff": float(np.mean(a - b)),
                "note": "identical results across seeds",
            }
            continue

        t, p = stats.ttest_rel(a, b)
        results[scenario] = {
            "valid": True,
            "t": float(t),
            "p": float(p),
            "dof": len(a) - 1,
            "significant": bool(p < 0.05),
            "mean_diff": float(np.mean(a - b)),
        }

    return results


def all_pairwise_tests(
    df: pd.DataFrame,
    policies: list[str],
    value_col: str = "normalized_fault_rate",
) -> pd.DataFrame:
    """Run paired t-tests for all policy pairs across all scenarios."""
    rows = []
    for i, policy_a in enumerate(policies):
        for policy_b in policies[i + 1:]:
            scenarios_results = paired_ttest_pairs(df, policy_a, policy_b, value_col)
            for scenario, r in scenarios_results.items():
                if not r.get("valid"):
                    continue
                rows.append({
                    "policy_a": policy_a,
                    "policy_b": policy_b,
                    "scenario": scenario,
                    "t": r["t"],
                    "p_value": r["p"],
                    "mean_diff": r["mean_diff"],
                    "significant": r["significant"],
                    "note": r.get("note", ""),
                })
    return pd.DataFrame(rows)


def failure_analysis(
    df: pd.DataFrame,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """For each policy, list scenarios where mean normalized fault rate exceeds threshold.

    A policy "fails" on a scenario if its mean norm rate > threshold (closer to LRU than to Belady).
    """
    summary = df.groupby(["policy", "scenario"])["normalized_fault_rate"].mean().reset_index()
    failures = summary[summary["normalized_fault_rate"] > threshold].copy()
    failures = failures.sort_values(["policy", "normalized_fault_rate"], ascending=[True, False])
    return failures


def render_charts(
    pivot: pd.DataFrame,
    output_dir: Path,
    df: pd.DataFrame,
) -> None:
    """Generate matplotlib charts."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Chart 1: per-scenario grouped bar chart (policies as bars within each scenario)
    fig, ax = plt.subplots(figsize=(14, 6))
    scenarios = pivot.columns.tolist()
    policies = pivot.index.tolist()
    x = np.arange(len(scenarios))
    width = 0.8 / max(len(policies), 1)

    for i, policy in enumerate(policies):
        values = pivot.loc[policy].to_numpy()
        ax.bar(x + i * width, values, width, label=policy)

    ax.set_xticks(x + width * (len(policies) - 1) / 2)
    ax.set_xticklabels(scenarios, rotation=20, ha="right")
    ax.set_ylabel("Normalized Fault Rate (0=Belady, 1=LRU)")
    ax.set_title("Eviction Policy Comparison Across SLM Workload Scenarios")
    ax.axhline(y=0.0, color="green", linestyle="--", alpha=0.3, label="_nolegend_")
    ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.3, label="_nolegend_")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    chart1 = output_dir / "policy_comparison_per_scenario.png"
    plt.savefig(chart1, dpi=120)
    plt.close()
    print(f"  Chart saved: {chart1}")

    # Chart 2: mean normalized fault rate per policy (overall)
    fig, ax = plt.subplots(figsize=(10, 5))
    means = pivot.mean(axis=1).sort_values()
    colors = ["green" if m < 0.3 else "orange" if m < 0.7 else "red" for m in means]
    ax.barh(means.index, means.values, color=colors)
    ax.set_xlabel("Mean Normalized Fault Rate (lower is better)")
    ax.set_title("Overall Policy Performance (mean across all scenarios)")
    for i, (policy, m) in enumerate(means.items()):
        ax.text(m + 0.02, i, f"{m:.3f}", va="center", fontsize=10)
    ax.set_xlim(0, max(means.values) * 1.15 + 0.1)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    chart2 = output_dir / "policy_overall_ranking.png"
    plt.savefig(chart2, dpi=120)
    plt.close()
    print(f"  Chart saved: {chart2}")

    # Chart 3: heatmap of (policy × scenario)
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(pivot.values, cmap="RdYlGn_r", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            v = pivot.values[i, j]
            color = "white" if v > 0.5 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color=color, fontsize=9)
    fig.colorbar(im, ax=ax, label="Normalized Fault Rate")
    ax.set_title("Normalized Fault Rate Heatmap (0=Belady, 1=LRU)")
    plt.tight_layout()
    chart3 = output_dir / "policy_heatmap.png"
    plt.savefig(chart3, dpi=120)
    plt.close()
    print(f"  Chart saved: {chart3}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze benchmark results")
    parser.add_argument("--input", default="data/results/benchmark_results.csv")
    parser.add_argument("--output-dir", default="data/results")
    parser.add_argument("--no-charts", action="store_true",
                        help="Skip chart generation")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} rows: "
          f"{df['policy'].nunique()} policies × "
          f"{df['scenario'].nunique()} scenarios × "
          f"{df['seed'].nunique()} seeds")

    # Pivot table
    print("\n=== Normalized Fault Rate (Policy × Scenario) ===")
    pivot = pivot_results(df)
    print(pivot.round(3).to_string())

    pivot_path = output_dir / "policy_scenario_matrix.csv"
    pivot.to_csv(pivot_path)
    print(f"\nPivot table saved: {pivot_path}")

    # Mean per policy
    print("\n=== Mean Normalized Fault Rate per Policy ===")
    means = pivot.mean(axis=1).sort_values()
    for policy, m in means.items():
        marker = "✓" if m < 0.3 else " "
        print(f"  {marker} {policy:20s} {m:.4f}")

    # Pairwise t-tests
    print("\n=== Pairwise Paired T-Tests (per scenario, across seeds) ===")
    policies = list(pivot.index)
    ttest_df = all_pairwise_tests(df, policies)
    ttest_path = output_dir / "pairwise_ttests.csv"
    ttest_df.to_csv(ttest_path, index=False)
    print(f"  Total comparisons: {len(ttest_df)}")
    sig = ttest_df[ttest_df["significant"]]
    print(f"  Statistically significant (p<0.05): {len(sig)}")

    # Show key comparisons: ML vs classical
    print("\nKey ML-vs-Classical comparisons:")
    for ml_policy in ["XGBoost", "MLP"]:
        for classical in ["LRU"]:
            sub = ttest_df[(ttest_df["policy_a"] == classical) &
                          (ttest_df["policy_b"] == ml_policy) |
                          ((ttest_df["policy_a"] == ml_policy) &
                           (ttest_df["policy_b"] == classical))]
            for _, row in sub.iterrows():
                a, b = row["policy_a"], row["policy_b"]
                diff = row["mean_diff"]
                p = row["p_value"]
                sig_marker = "**" if row["significant"] else "  "
                print(f"  {sig_marker} {a} vs {b}, {row['scenario']}: "
                      f"mean_diff={diff:+.3f}, p={p:.4f}")

    # Failure analysis
    print("\n=== Failure Analysis (norm_rate > 0.5) ===")
    failures = failure_analysis(df, threshold=0.5)
    if len(failures) == 0:
        print("  No policy fails on any scenario at threshold 0.5")
    else:
        for policy in sorted(failures["policy"].unique()):
            scenarios = failures[failures["policy"] == policy]
            scenario_list = ", ".join(
                f"{r['scenario']}({r['normalized_fault_rate']:.2f})"
                for _, r in scenarios.iterrows()
            )
            print(f"  {policy}: {scenario_list}")
    failures_path = output_dir / "failure_analysis.csv"
    failures.to_csv(failures_path, index=False)
    print(f"\nFailure analysis saved: {failures_path}")

    # Charts
    if not args.no_charts:
        print("\n=== Generating charts ===")
        try:
            render_charts(pivot, output_dir, df)
        except Exception as e:
            print(f"  Chart generation skipped: {e}")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
