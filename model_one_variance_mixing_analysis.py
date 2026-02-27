"""
Plot Model One variance and spectral mixing time versus lambda.

This script reads:
- Stationary stats from:
    outputs/model1/stationary/model1_stationary_stats.txt
- Spectral mixing-time proxies from:
    outputs/model1/mixing/model1_mixing_spectral.txt

and produces a plot with variance(λ) and t_mix_spec(λ).
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


def load_mixing_time(path: str = MIXING_STATS_PATH):
    lambdas = []
    t_mix = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "lambda" or parts[0].startswith("-"):
                continue
            lam, _lambda2, _gap, t_spec = parts
            lambdas.append(float(lam))
            t_mix.append(float(t_spec))
    return np.array(lambdas), np.array(t_mix)


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


if __name__ == "__main__":
    plot_variance_and_mixing()
