"""
Plot variance and empirical mixing time for Model Two from a saved summary file.
"""

import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

def load_lambda_sweep_summary(
    summary_path: str,
) -> list[tuple[float, float, float, int, int]]:
    """Load lambda-sweep rows from the summary file."""
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
    return rows


def plot_variance_and_mixing(summary_path: str, plot_path: str, n_agents: int) -> None:
    """Plot variance and empirical mixing time versus lambda from the summary file."""
    rows = load_lambda_sweep_summary(summary_path)
    if not rows:
        raise ValueError("No lambda-sweep rows were found in the summary file.")

    lam_vals = np.array([row[0] for row in rows])
    var_vals = np.array([row[1] for row in rows])
    t_mix_vals = np.array([row[3] for row in rows])
    min_var_index = int(np.argmin(var_vals))
    max_t_mix_index = int(np.argmax(t_mix_vals))
    min_var_lambda = lam_vals[min_var_index]
    max_t_mix_lambda = lam_vals[max_t_mix_index]

    fig, ax1 = plt.subplots()
    ax1.set_xlabel(r"$\lambda$")
    ax1.set_ylabel("Variance of final outcome", color="tab:blue")
    ax1.plot(lam_vals, var_vals, color="tab:blue", label="variance")
    ax1.scatter(
        [min_var_lambda],
        [var_vals[min_var_index]],
        color="tab:blue",
        s=20,
        zorder=2,
    )
    ax1.axvline(
        min_var_lambda,
        color="tab:blue",
        linestyle=":",
        linewidth=1.5,
        alpha=0.7,
    )
    ax1.annotate(
        rf"$\lambda={min_var_lambda:.2f}$",
        xy=(min_var_lambda, var_vals[min_var_index]),
        xytext=(8, 8),
        textcoords="offset points",
        color="tab:blue",
    )
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.set_ylabel(r"$t_{\mathrm{mix}}$ (empirical)", color="tab:red")
    ax2.plot(lam_vals, t_mix_vals, color="tab:red", label="t_mix")
    ax2.scatter(
        [max_t_mix_lambda],
        [t_mix_vals[max_t_mix_index]],
        color="tab:red",
        s=20,
        zorder=2,
    )
    ax2.axvline(
        max_t_mix_lambda,
        color="tab:red",
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
    )
    ax2.annotate(
        rf"$\lambda={max_t_mix_lambda:.2f}$",
        xy=(max_t_mix_lambda, t_mix_vals[max_t_mix_index]),
        xytext=(8, -14),
        textcoords="offset points",
        color="tab:red",
    )
    ax2.tick_params(axis="y", labelcolor="tab:red")

    fig.suptitle(f"Model Two: variance and mixing time vs lambda (n={n_agents})")
    fig.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()

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

    base_output_dir = os.path.join(os.path.dirname(__file__), "outputs", "model2")
    output_dir = os.path.join(base_output_dir, f"{n_agents}nodes")

    summary_path = os.path.join(output_dir, "model2_variance_mixing_vs_lambda.txt")
    plot_path = os.path.join(output_dir, "model2_variance_mixing_vs_lambda.png")

    plot_variance_and_mixing(summary_path, plot_path, n_agents)