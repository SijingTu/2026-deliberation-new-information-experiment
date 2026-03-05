"""
Plot Model One variance and spectral mixing time versus lambda.

This script reads (under outputs/model1/{n}nodes/, matching model2 structure):
- Stationary stats from:
    outputs/model1/{n}nodes/stationary/model1_stationary_stats.txt
- Spectral mixing-time proxies from:
    outputs/model1/{n}nodes/mixing/model1_mixing_spectral.txt

and produces under outputs/model1/{n}nodes/analysis/:
- a plot with variance(λ) and t_mix_spec(λ)
- a plot comparing t_mix_spec(λ) to reciprocal conjectures based on
  spectral_gap(λ) ≈ a (1 - λ)
"""

import argparse
import os
import matplotlib.pyplot as plt
import numpy as np


BASE_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs", "model1")
DEFAULT_N = 1000


def _nodes_dir(n: int = DEFAULT_N) -> str:
    """Return outputs/model1/{n}nodes, matching model2 folder structure."""
    return os.path.join(BASE_OUTPUT_DIR, f"{n}nodes")


STATIONARY_STATS_PATH = os.path.join(
    _nodes_dir(), "stationary", "model1_stationary_stats.txt"
)
MIXING_STATS_PATH = os.path.join(
    _nodes_dir(), "mixing", "model1_mixing_spectral.txt"
)
ANALYSIS_DIR = os.path.join(_nodes_dir(), "analysis")


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


def variance_conjecture(lam: float | np.ndarray, n: int) -> float | np.ndarray:
    r"""
    Original conjectured variance (for reference):

    .. math::
        \sigma^2(\lambda) = \frac{(n-1)^2 (1-\lambda)}{12(3+\lambda)}
    """
    lam = np.asarray(lam)
    return (n - 1) ** 2 * (1.0 - lam) / (12.0 * (3.0 + lam))


def variance_model(
    lam: float | np.ndarray, n: int, a: float, b: float
) -> float | np.ndarray:
    r"""
    Parametric model for variance: σ²(λ) = (n−1)²(1−λ) / (a + bλ).
    Parameters a and b are fitted from data.
    """
    lam = np.asarray(lam)
    denom = a + b * lam
    if np.any(denom <= 0):
        raise ValueError("a + b*λ must be positive for all λ in the range.")
    return ((n - 1) ** 2) * (1.0 - lam) / denom


def fit_variance_model(
    lam: np.ndarray, variances: np.ndarray, n: int
) -> tuple[float, float]:
    r"""
    Fit parameters a, b in σ²(λ) = (n−1)²(1−λ)/(a + bλ) from empirical (lam, variances).

    From the model, a + b·λ = (n−1)²(1−λ)/σ², so we solve the linear system
    [1, λ_i] @ [a, b] = (n−1)²(1−λ_i)/σ²_i in least-squares sense.
    """
    valid = variances > 1e-15
    if not np.any(valid):
        raise ValueError("No positive variances to fit.")
    lam_f = lam[valid]
    var_f = variances[valid]
    y = ((n - 1) ** 2) * (1.0 - lam_f) / var_f
    X = np.column_stack([np.ones_like(lam_f), lam_f])
    (a, b), *_ = np.linalg.lstsq(X, y, rcond=None)
    if a + b * np.min(lam_f) <= 0 or a + b * np.max(lam_f) <= 0:
        raise ValueError(
            "Fitted a, b give non-positive denominator in the λ range."
        )
    return float(a), float(b)


def plot_variance_with_conjecture(n: int = DEFAULT_N) -> None:
    """
    Side-by-side: (left) data vs fitted (n−1)²(1−λ)/(a+bλ); (right) data vs
    original conjecture (n−1)²(1−λ)/(12(3+λ)).
    """
    nodes_dir = _nodes_dir(n)
    analysis_dir = os.path.join(nodes_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    stationary_path = os.path.join(
        nodes_dir, "stationary", "model1_stationary_stats.txt"
    )
    lam_var, variances = load_stationary_variance(stationary_path)
    a, b = fit_variance_model(lam_var, variances, n)
    var_fitted = variance_model(lam_var, n, a, b)
    var_original = variance_conjecture(lam_var, n)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    # Left: data vs fitted (n-1)²(1-λ)/(a+bλ)
    ax_left = axes[0]
    ax_left.plot(
        lam_var,
        variances,
        color="tab:blue",
        linewidth=2.0,
        label="Data",
    )
    ax_left.plot(
        lam_var,
        var_fitted,
        color="tab:orange",
        linestyle="--",
        linewidth=2.0,
        label=rf"Fitted $(n-1)^2(1-\lambda)/(a+b\lambda)$, $a={a:.2f}$, $b={b:.2f}$",
    )
    ax_left.set_xlabel(r"$\lambda$")
    ax_left.set_ylabel("Variance")
    ax_left.set_title(r"Data vs fitted $(n-1)^2(1-\lambda)/(a+b\lambda)$")
    ax_left.legend()
    ax_left.grid(True, alpha=0.3)

    # Right: data vs original conjecture (n-1)²(1-λ)/(12(3+λ))
    ax_right = axes[1]
    ax_right.plot(
        lam_var,
        variances,
        color="tab:blue",
        linewidth=2.0,
        label="Data",
    )
    ax_right.plot(
        lam_var,
        var_original,
        color="tab:green",
        linestyle="--",
        linewidth=2.0,
        label=r"Original $\frac{(n-1)^2(1-\lambda)}{12(3+\lambda)}$",
    )
    ax_right.set_xlabel(r"$\lambda$")
    ax_right.set_ylabel("Variance")
    ax_right.set_title(r"Data vs original conjecture $\frac{(n-1)^2(1-\lambda)}{12(3+\lambda)}$")
    ax_right.legend()
    ax_right.grid(True, alpha=0.3)

    fig.tight_layout()
    output_path = os.path.join(
        analysis_dir, "model1_variance_vs_conjecture.png"
    )
    fig.savefig(output_path)
    plt.close(fig)


def fit_gap_linear_through_origin(lambdas: np.ndarray, gaps: np.ndarray) -> float:
    one_minus_lambda = 1.0 - lambdas
    denominator = float(np.dot(one_minus_lambda, one_minus_lambda))
    if denominator <= 0.0:
        raise ValueError("Need at least one lambda with 1 - lambda > 0.")
    return float(np.dot(one_minus_lambda, gaps) / denominator)


def plot_variance_and_mixing(n: int = DEFAULT_N):
    nodes_dir = _nodes_dir(n)
    analysis_dir = os.path.join(nodes_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    stationary_path = os.path.join(
        nodes_dir, "stationary", "model1_stationary_stats.txt"
    )
    mixing_path = os.path.join(nodes_dir, "mixing", "model1_mixing_spectral.txt")

    lam_var, variances = load_stationary_variance(stationary_path)
    lam_mix, t_mix = load_mixing_time(mixing_path)

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
    output_path = os.path.join(
        analysis_dir, "model1_variance_mixing_vs_lambda.png"
    )
    fig.savefig(output_path)
    plt.close(fig)


def plot_mixing_with_conjecture(
    eps_mixing: float = 1e-3, n: int = DEFAULT_N
) -> None:
    if not (0.0 < eps_mixing < 1.0):
        raise ValueError("eps_mixing must be in (0, 1).")

    nodes_dir = _nodes_dir(n)
    analysis_dir = os.path.join(nodes_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    mixing_path = os.path.join(nodes_dir, "mixing", "model1_mixing_spectral.txt")
    lambdas, _lambda2, gaps, t_mix = load_mixing_spectral(mixing_path)
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

    output_path = os.path.join(
        analysis_dir, "model1_mixing_vs_lambda_conjecture.png"
    )
    fig.savefig(output_path)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot Model One variance and spectral mixing time versus lambda."
    )
    parser.add_argument(
        "n",
        type=int,
        nargs="?",
        default=DEFAULT_N,
        help=f"Number of nodes (default: {DEFAULT_N})",
    )
    args = parser.parse_args()
    n = args.n
    plot_variance_and_mixing(n=n)
    plot_variance_with_conjecture(n=n)
    plot_mixing_with_conjecture(n=n)
