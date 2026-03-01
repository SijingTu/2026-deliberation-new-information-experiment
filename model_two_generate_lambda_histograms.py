"""
Generate Model Two histogram data across a lambda sweep.

This module runs a lambda sweep using the simulation core in
`model_two_monte_carlo.py`, writes summary rows immediately after each lambda
finishes, regenerates histogram plots from the saved histogram data, and uses
the separate variance/mixing plot module for the summary plot.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from model_two_monte_carlo import estimate_empirical_mixing_time, run_model_two_simulation
# from model_two_varaince_mixing_plot import plot_variance_and_mixing


def initialize_lambda_sweep_summary(summary_path: str) -> None:
    """Create or overwrite the lambda-sweep summary file with its header."""
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as summary_file:
        summary_file.write("lambda variance std t_mix n_rounds\n")


def append_lambda_sweep_result(
    summary_path: str,
    lam: float,
    variance: float,
    std_dev: float,
    t_mix: int,
    n_rounds: int,
) -> None:
    """Append one completed lambda result to the summary file."""
    with open(summary_path, "a", encoding="utf-8") as summary_file:
        summary_file.write(
            f"{lam:.4f} {variance:.6f} {std_dev:.6f} {t_mix} {n_rounds}\n"
        )


def load_histogram_data(
    histogram_data_path: str,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    """Load saved histogram stats plus per-outcome probabilities from a data file."""
    with open(histogram_data_path, "r", encoding="utf-8") as histogram_file:
        stats_header = histogram_file.readline().split()
        stats_values = histogram_file.readline().split()
        counts_header = histogram_file.readline().split()

        if stats_header != ["n_agents", "n_rounds", "n_runs", "lam", "mean", "variance", "std"]:
            raise ValueError(f"Unexpected stats header in {histogram_data_path}.")
        if counts_header != ["outcome", "count", "fraction"]:
            raise ValueError(f"Unexpected histogram header in {histogram_data_path}.")
        if len(stats_values) != len(stats_header):
            raise ValueError(f"Malformed stats row in {histogram_data_path}.")

        stats = {
            "n_agents": int(stats_values[0]),
            "n_rounds": int(stats_values[1]),
            "n_runs": int(stats_values[2]),
            "lam": float(stats_values[3]),
            "mean": float(stats_values[4]),
            "variance": float(stats_values[5]),
            "std": float(stats_values[6]),
        }

        outcomes = []
        fractions = []
        for line in histogram_file:
            parts = line.split()
            if not parts:
                continue
            outcome, _count, fraction = parts
            outcomes.append(int(outcome))
            fractions.append(float(fraction))

    if not outcomes:
        raise ValueError(f"No histogram rows were found in {histogram_data_path}.")

    return stats, np.array(outcomes), np.array(fractions)


def plot_histogram_from_data(histogram_data_path: str, plot_path: str) -> None:
    """Draw one histogram from a saved histogram-data file."""
    stats, outcomes, fractions = load_histogram_data(histogram_data_path)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(outcomes, fractions, alpha=0.7, color="steelblue", edgecolor="black")
    ax.axvline(
        (stats["n_agents"] + 1) / 2,
        color="red",
        linestyle="--",
        linewidth=2,
        label="Median position",
    )
    ax.axvline(
        stats["mean"],
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Mean={stats['mean']:.2f}",
    )
    ax.set_xlabel("Final Outcome Position")
    ax.set_ylabel("Probability")
    ax.set_title(
        "Model Two: Final Outcome "
        f"(n={stats['n_agents']}, T={stats['n_rounds']}, "
        f"lambda={stats['lam']:.2f}, Var={stats['variance']:.2f})"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_histograms_from_directory(
    histogram_data_dir: str,
    histogram_plots_dir: str,
) -> list[str]:
    """Draw saved histogram files one by one from a directory of histogram data."""
    os.makedirs(histogram_plots_dir, exist_ok=True)

    plot_paths = []
    for file_name in sorted(os.listdir(histogram_data_dir)):
        if not file_name.endswith(".txt"):
            continue
        histogram_data_path = os.path.join(histogram_data_dir, file_name)
        plot_name = f"{os.path.splitext(file_name)[0]}.png"
        plot_path = os.path.join(histogram_plots_dir, plot_name)
        plot_histogram_from_data(histogram_data_path, plot_path)
        plot_paths.append(plot_path)

    return plot_paths


def run_model_two_lambda_sweep(
    n_agents: int = 50,
    lambdas=None,
    time_grid=None,
    n_runs: int = 1000,
    T_ref: int = 5000,
    n_chains_mixing: int = 500,
    mixing_multiplier: int = 2,
    T_min: int = 500,
    eps_mixing: float = 0.1,
    seed: int = 0,
    output_dir: str = "",
    summary_path: str = "",
    histograms_dir: str = "",
):
    """
    Run Model Two over a grid of lambdas and persist each result immediately.

    For each lambda, estimate t_mix, derive n_rounds, run the simulation,
    append one summary row, and save per-lambda histogram data. Plot generation
    is separate and should call `plot_variance_and_mixing(...)` after the sweep.
    """
    if not output_dir or not summary_path or not histograms_dir:
        raise ValueError(
            "output_dir, summary_path, and histograms_dir must all be provided."
        )

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(histograms_dir, exist_ok=True)

    lambda_values = np.atleast_1d(np.linspace(0, 0.99, 20) if lambdas is None else lambdas)
    if time_grid is None:
        raise ValueError("time_grid must be provided.")
    time_grid = sorted(set(time_grid) | {T_ref})

    initialize_lambda_sweep_summary(summary_path)

    for lam_value in lambda_values:
        lam = float(lam_value)
        t_mix, _ = estimate_empirical_mixing_time(
            n_agents,
            T_ref,
            n_chains_mixing,
            time_grid=time_grid,
            eps=eps_mixing,
            lam=lam,
            seed=seed,
        )
        n_rounds = max(T_min, int(mixing_multiplier * t_mix))
        histogram_path = os.path.join(
            histograms_dir, f"model2_histogram_lambda_{lam:.2f}.txt"
        )
        result = run_model_two_simulation(
            n_agents,
            n_runs,
            n_rounds=n_rounds,
            T_ref=T_ref,
            n_chains_mixing=n_chains_mixing,
            mixing_multiplier=mixing_multiplier,
            T_min=T_min,
            eps_mixing=eps_mixing,
            lam=lam,
            seed=seed,
            output_dir=output_dir,
            histogram_path=histogram_path,
        )
        append_lambda_sweep_result(
            summary_path,
            lam,
            result["variance"],
            result["std"],
            t_mix,
            n_rounds,
        )
        print(
            f"lambda={lam:.2f}: t_mix={t_mix}, n_rounds={n_rounds}, "
            f"var={result['variance']:.4f}, std={result['std']:.4f}"
        )

    rows = []
    with open(summary_path, "r", encoding="utf-8") as summary_file:
        for line in summary_file:
            parts = line.split()
            if not parts or parts[0] == "lambda":
                continue
            lam, variance, std_dev, t_mix, n_rounds = parts
            rows.append(
                (
                    float(lam),
                    float(variance),
                    float(std_dev),
                    int(t_mix),
                    int(n_rounds),
                )
            )
    return rows, summary_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the Model Two lambda sweep and regenerate histogram files."
    )
    parser.add_argument(
        "n_agents",
        type=int,
        help="Number of agents for Model Two.",
    )
    args = parser.parse_args()

    n_agents = args.n_agents
    lambdas = [i * 0.01 for i in range(100)]
    time_grid = [10, 20, 30, 40, 50, 100, 200, 500, 1000, 2000, 3000, 4000, 5000]
    base_output_dir = os.path.join(os.path.dirname(__file__), "outputs", "model2")
    output_dir = os.path.join(base_output_dir, f"{n_agents}nodes")
    histograms_data_dir = os.path.join(output_dir, "histogram_data")
    histograms_plots_dir = os.path.join(output_dir, "histograms")
    summary_path = os.path.join(output_dir, "model2_variance_mixing_vs_lambda.txt")
    # plot_path = os.path.join(output_dir, "model2_variance_mixing_vs_lambda.png")

    # Note, change the number of chains for the mixing time estimation and the number of runs for different n_agents.
    rows, summary_path = run_model_two_lambda_sweep(
        n_agents=n_agents,
        lambdas=lambdas,
        time_grid=time_grid,
        n_runs=500,
        T_ref=500,
        n_chains_mixing=500,
        mixing_multiplier=2,
        T_min=50,
        eps_mixing=0.1,
        seed=0,
        output_dir=output_dir,
        summary_path=summary_path,
        histograms_dir=histograms_data_dir,
    )

    histogram_plot_paths = plot_histograms_from_directory(
        histograms_data_dir, histograms_plots_dir
    )
    # plot_variance_and_mixing(summary_path, plot_path, n_agents)

    print(f"Completed {len(rows)} lambda values.")
    print(f"Summary written to {summary_path}")
    print(f"Histogram plots written: {len(histogram_plot_paths)}")
    # print(f"Variance/mixing plot saved to {plot_path}")
