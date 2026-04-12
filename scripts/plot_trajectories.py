#!/usr/bin/env python3
"""Plot CACHEUS expert weight trajectories from tune_cacheus.py output.

Reads cacheus_trajectories.json (one entry per scenario) and produces:
- One chart per scenario showing expert weights over time
- A combined chart showing convergence across scenarios

Usage:
    python scripts/plot_trajectories.py --input data/results/cacheus_trajectories.json \\
                                        --output-dir data/results/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def plot_scenario_trajectory(traj: dict, output_path: Path) -> None:
    """Plot one scenario's weight trajectory."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ticks = np.array(traj["ticks"])
    weights = np.array(traj["weights"])  # (num_decisions, num_experts)
    expert_names = traj.get("expert_names", [f"expert_{i}" for i in range(weights.shape[1])])
    phase_changes = traj.get("phase_changes", [])

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, name in enumerate(expert_names):
        ax.plot(ticks, weights[:, i], label=name, linewidth=1.5)

    # Mark phase change events
    for tick in phase_changes:
        ax.axvline(x=tick, color="gray", linestyle=":", alpha=0.5)

    ax.set_xlabel("Simulation tick")
    ax.set_ylabel("Expert weight")
    ax.set_title(f"CACHEUS Expert Weights Over Time — {traj['scenario']}")
    ax.set_ylim(0, 1)
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()
    print(f"  Plot: {output_path}")


def plot_all_scenarios_grid(
    trajectories: dict, output_path: Path,
) -> None:
    """Plot all scenarios in a single multi-panel figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(trajectories)
    cols = 2
    rows = (n + 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(14, 4 * rows))
    axes = axes.flatten() if n > 1 else [axes]

    for idx, (scenario_name, traj) in enumerate(trajectories.items()):
        ax = axes[idx]
        ticks = np.array(traj["ticks"])
        if len(ticks) == 0:
            ax.text(0.5, 0.5, "No decisions recorded",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(scenario_name)
            continue

        weights = np.array(traj["weights"])
        expert_names = traj.get("expert_names", [])

        for i, name in enumerate(expert_names):
            ax.plot(ticks, weights[:, i], label=name, linewidth=1.2)

        for t in traj.get("phase_changes", []):
            ax.axvline(x=t, color="gray", linestyle=":", alpha=0.4)

        ax.set_title(f"{scenario_name} ({len(ticks)} decisions)", fontsize=11)
        ax.set_xlabel("tick")
        ax.set_ylabel("weight")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        if idx == 0:
            ax.legend(loc="best", fontsize=8)

    # Hide unused subplots
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close()
    print(f"  Combined plot: {output_path}")


def adaptation_speed(traj: dict, threshold: float = 0.05) -> int | None:
    """Estimate ticks until weights stop changing significantly.

    Returns the tick index at which the L1 weight change between consecutive
    decisions falls below `threshold` (i.e., adaptation has converged).
    """
    weights = np.array(traj["weights"])
    if len(weights) < 2:
        return None

    diffs = np.abs(np.diff(weights, axis=0)).sum(axis=1)
    converged = np.where(diffs < threshold)[0]
    if len(converged) == 0:
        return None
    return int(traj["ticks"][converged[0]])


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot CACHEUS weight trajectories")
    parser.add_argument(
        "--input", default="data/results/cacheus_trajectories.json",
    )
    parser.add_argument("--output-dir", default="data/results")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} not found. Run tune_cacheus.py first.")
        return

    with open(input_path) as f:
        trajectories = json.load(f)

    print(f"Loaded {len(trajectories)} scenario trajectories")

    # Per-scenario plots
    for scenario_name, traj in trajectories.items():
        traj["scenario"] = scenario_name
        if len(traj.get("ticks", [])) == 0:
            print(f"  Skipping {scenario_name}: no decisions recorded")
            continue
        out = output_dir / f"trajectory_{scenario_name}.png"
        plot_scenario_trajectory(traj, out)

    # Combined plot
    combined = output_dir / "trajectory_all_scenarios.png"
    plot_all_scenarios_grid(trajectories, combined)

    # Adaptation speed analysis
    print("\n=== Adaptation Speed (ticks until weight change < 0.05) ===")
    for name, traj in trajectories.items():
        traj["scenario"] = name
        speed = adaptation_speed(traj)
        if speed is None:
            print(f"  {name}: no convergence detected")
        else:
            print(f"  {name}: converged at tick {speed}")


if __name__ == "__main__":
    main()
