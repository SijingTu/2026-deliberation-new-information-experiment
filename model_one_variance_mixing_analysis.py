"""
Plot Model One variance and spectral mixing time versus lambda.

This script reads:
- Stationary stats from:
    outputs/model1/stationary/model1_stationary_stats.txt
- Spectral mixing-time proxies from:
    outputs/model1/mixing/model1_mixing_spectral.txt

and produces:
- a plot with variance(λ) and t_mix_spec(λ)
- a plot comparing t_mix_spec(λ) to reciprocal conjectures based on
  spectral_gap(λ) ≈ a (1 - λ)
"""

import os
import matplotlib.pyplot as plt
import numpy as np


BASE_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs", "model1")
STATIONARY_STATS_PATH = os.path.join(
    BASE_OUTPUT_DIR, "stationary", "model1_stationary_stats.txt"
)
MIXING_STATS_PATH = os.path.join(
    BASE_OUTPUT_DIR, "mixing", "model1_mixing_spectral.txt"
)
ANALYSIS_DIR = os.path.join(BASE_OUTPUT_DIR, "analysis")


def load_stationary_variance(path: str = STATIONARY_STATS_PATH):
    lambdas = []
    variances = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            # Skip header or separator lines
            if not parts:
                continue
            if parts[0] == "lambda" or parts[0].startswith("-"):
                continue
            lam, _mean, var, _std = parts
            lambdas.append(float(lam))
            variances.append(float(var))
    return np.array(lambdas), np.array(variances)


def load_mixing_spectral(path: str = MIXING_STATS_PATH):
    lambdas = []
    lambda2_vals = []
    gaps = []
    t_mix = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "lambda" or parts[0].startswith("-"):
                continue
            lam, lambda2, gap, t_spec = parts
            lambdas.append(float(lam))
            lambda2_vals.append(float(lambda2))
            gaps.append(float(gap))
            t_mix.append(float(t_spec))
    return (
        np.array(lambdas),
        np.array(lambda2_vals),
        np.array(gaps),
        np.array(t_mix),
    )


def load_mixing_time(path: str = MIXING_STATS_PATH):
    lambdas, _lambda2, _gaps, t_mix = load_mixing_spectral(path)
    return lambdas, t_mix


def fit_gap_linear_through_origin(lambdas: np.ndarray, gaps: np.ndarray) -> float:
    one_minus_lambda = 1.0 - lambdas
    denominator = float(np.dot(one_minus_lambda, one_minus_lambda))
    if denominator <= 0.0:
        raise ValueError("Need at least one lambda with 1 - lambda > 0.")
    return float(np.dot(one_minus_lambda, gaps) / denominator)


def plot_variance_and_mixing():
    os.makedirs(ANALYSIS_DIR, exist_ok=True)

    lam_var, variances = load_stationary_variance()
    lam_mix, t_mix = load_mixing_time()

    fig, ax1 = plt.subplots()

    ax1.set_xlabel(r"$\lambda$")
    ax1.set_ylabel("Variance of stationary distribution", color="tab:blue")
    ax1.plot(lam_var, variances, color="tab:blue", label="variance")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.set_ylabel(r"$t_{\mathrm{mix}}^{(\mathrm{spec})}$", color="tab:red")
    ax2.plot(lam_mix, t_mix, color="tab:red", label="mixing time (spectral)")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    fig.tight_layout()
    output_path = os.path.join(ANALYSIS_DIR, "model1_variance_mixing_vs_lambda.png")
    fig.savefig(output_path)
    plt.close(fig)


def plot_mixing_with_conjecture(eps_mixing: float = 1e-3) -> None:
    if not (0.0 < eps_mixing < 1.0):
        raise ValueError("eps_mixing must be in (0, 1).")

    os.makedirs(ANALYSIS_DIR, exist_ok=True)

    lambdas, _lambda2, gaps, t_mix = load_mixing_spectral()
    a_fit = fit_gap_linear_through_origin(lambdas, gaps)
    one_minus_lambda = 1.0 - lambdas

    g_fit = a_fit * one_minus_lambda
    g_half = 0.5 * one_minus_lambda
    log_term = np.log(1.0 / eps_mixing)

    with np.errstate(divide="ignore", invalid="ignore"):
        t_fit = np.where(g_fit > 0.0, log_term / g_fit, np.nan)
        t_half = np.where(g_half > 0.0, log_term / g_half, np.nan)

    valid_fit = np.isfinite(t_fit) & (t_fit > 0.0)
    valid_half = np.isfinite(t_half) & (t_half > 0.0)
    valid_obs = np.isfinite(t_mix) & (t_mix > 0.0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)

    for ax in axes:
        ax.plot(
            lambdas[valid_obs],
            t_mix[valid_obs],
            color="tab:red",
            linewidth=2.0,
            label=r"observed $t_{\mathrm{mix}}^{(\mathrm{spec})}$",
        )
        ax.plot(
            lambdas[valid_fit],
            t_fit[valid_fit],
            color="tab:blue",
            linestyle="--",
            linewidth=2.0,
            label=rf"conjecture $a_{{\mathrm{{fit}}}}={a_fit:.4f}$",
        )
        ax.plot(
            lambdas[valid_half],
            t_half[valid_half],
            color="tab:green",
            linestyle=":",
            linewidth=2.5,
            label=r"conjecture $a=0.5$",
        )
        ax.set_xlabel(r"$\lambda$")
        ax.grid(True, alpha=0.25)

    axes[0].set_ylabel(r"$t_{\mathrm{mix}}$")
    axes[0].set_title("Linear Scale")
    axes[1].set_title("Log Scale")
    axes[1].set_yscale("log")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))

    output_path = os.path.join(ANALYSIS_DIR, "model1_mixing_vs_lambda_conjecture.png")
    fig.savefig(output_path)
    plt.close(fig)


if __name__ == "__main__":
    plot_variance_and_mixing()
    plot_mixing_with_conjecture()
