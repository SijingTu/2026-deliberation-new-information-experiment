"""
Model One: Markov chain on state space {1, 2, ..., n} and its stationary distribution.

We follow the model in 020modelone.tex. There are n agents, each with a bliss point
a*_u in {1, 2, ..., n}, drawn i.i.d. uniformly. Stances and outcomes lie in [1, n].
At deliberation round t, the state X_t is the current outcome o_t.

At each round:
- Sample two agents u and v uniformly at random.
- Given the previous outcome X_{t-1}, form temporary stances
    a_u^t = a*_u (1 - λ) + X_{t-1} λ,
    a_v^t = a*_v (1 - λ) + X_{t-1} λ,
  where λ in (0, 1) controls how responsive stances are to the previous outcome.
- The new outcome is the median of the two stances and the previous outcome:
    X_t = median(a_u^t, a_v^t, X_{t-1}).

With bliss points i.i.d. uniform on {1, 2, ..., n}, this defines a Markov chain (X_t)
with conditional CDF H_i(z) = P(X_t <= z | X_{t-1} = i). We discretize using
intervals I_j = (j - 1/2, j + 1/2] for j = 1, ..., n, so the transition matrix is
P_{i,j} = H_i(j + 1/2) - H_i(j - 1/2). The stationary distribution pi is computed
by power iteration on P.
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt


BASE_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs", "model1")
STATIONARY_DIR = os.path.join(BASE_OUTPUT_DIR, "stationary")
MIXING_DIR = os.path.join(BASE_OUTPUT_DIR, "mixing")


def _H_i(z: float, i_val: int, lam: float, n: int) -> float:
    """
    Transition CDF H_i(z) = P(X_t <= z | X_{t-1} = i).

    State value i is in {1, ..., n}. Formula from 020modelone.tex eq (91)-(98):
    - 0 if z < a_i
    - (1/n * floor((z - λi)/(1-λ)))^2 if a_i <= z < i
    - 1 - (1 - 1/n*floor(...))^2 if i <= z < b_i
    - 1 if z >= b_i
    where a_i = λi + 1 - λ, b_i = (1-λ)n + λi.
    """
    if lam >= 1.0:
        raise ValueError("lambda must be in [0, 1).")
    a_i = lam * i_val + 1.0 - lam
    b_i = (1.0 - lam) * n + lam * i_val
    if z < a_i:
        return 0.0
    if z >= b_i:
        return 1.0
    denom = 1.0 - lam
    # floor((z - λi)/(1-λ)); in [a_i, b_i) we have 1 <= floor <= n
    raw = (z - lam * i_val) / denom
    k = int(np.floor(raw))
    k = max(1, min(n, k))
    frac = k / n
    if z < i_val:
        return frac * frac
    else:
        return 1.0 - (1.0 - frac) * (1.0 - frac)


def build_transition_matrix(lam: float, n: int, renormalize: bool = True):
    """
    Build the n×n transition matrix P for Model One with discrete bliss points.

    State index 0 corresponds to value 1, ..., state index n-1 to value n.
    Interval I_j = (j - 1/2, j + 1/2] for j = 1, ..., n.
    P[i, j] = H_{i+1}((j+1) + 1/2) - H_{i+1}((j+1) - 1/2).
    """
    if not (0 <= lam < 1):
        raise ValueError("lambda must be in [0, 1).")

    P = np.zeros((n, n), dtype=float)
    for i_idx in range(n):
        i_val = i_idx + 1
        for j_idx in range(n):
            j_val = j_idx + 1
            ell_j = j_val - 0.5
            u_j = j_val + 0.5
            P[i_idx, j_idx] = _H_i(u_j, i_val, lam, n) - _H_i(ell_j, i_val, lam, n)

        if renormalize:
            row_sum = P[i_idx].sum()
            if row_sum > 0:
                P[i_idx] /= row_sum

    bins = np.linspace(0.5, n + 0.5, n + 1)
    return P, bins


def power_iteration_stationary(P: np.ndarray, eps: float = 1e-10, max_iter: int = 1_000_000):
    """
    Power iteration on row-stochastic P to find stationary distribution pi.
    """
    n_states = P.shape[0]
    pi = np.ones(n_states) / n_states

    for _ in range(max_iter):
        pi_new = pi @ P
        if np.linalg.norm(pi_new - pi, ord=1) < eps:
            return pi_new
        pi = pi_new

    return pi  # return last iterate if not converged


def simulate(lam: float, n: int = 2000, eps: float = 1e-10, renormalize: bool = True):
    P, bins = build_transition_matrix(lam, n, renormalize=renormalize)
    pi = power_iteration_stationary(P, eps=eps)
    return pi, bins


def draw_histogram(
    pi: np.ndarray,
    bins: np.ndarray,
    output_path: str = "stationary_distribution.png",
    title: str = "Stationary distribution",
) -> None:
    if bins.shape[0] != pi.shape[0] + 1:
        raise ValueError("bins must be length len(pi) + 1.")

    centers = 0.5 * (bins[:-1] + bins[1:])
    width = bins[1] - bins[0]
    fig, ax = plt.subplots()
    ax.bar(centers, pi, width=width, align="center", edgecolor="none")
    ax.set_xlabel("x (bin midpoint)")
    ax.set_ylabel("pi")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def distribution_stats(pi: np.ndarray, bins: np.ndarray) -> tuple[float, float, float]:
    if bins.shape[0] != pi.shape[0] + 1:
        raise ValueError("bins must be length len(pi) + 1.")

    weights = pi / pi.sum()
    centers = 0.5 * (bins[:-1] + bins[1:])
    mean = np.sum(weights * centers)
    variance = np.sum(weights * (centers - mean) ** 2)
    std_dev = np.sqrt(variance)
    return mean, variance, std_dev


def write_all_stats_to_file(data_dir: str | None = None) -> None:
    """
    Gather statistics from all histogram data files in the given directory
    and write them to a table (model1_stationary_stats.txt) under the same directory.
    """
    target_dir = data_dir or STATIONARY_DIR
    os.makedirs(target_dir, exist_ok=True)
    stats_path = os.path.join(target_dir, "model1_stationary_stats.txt")

    rows = []
    for fname in sorted(os.listdir(target_dir)):
        if not fname.endswith(".txt") or fname == "model1_stationary_stats.txt":
            continue
        data_path = os.path.join(target_dir, fname)
        try:
            stats, _, _ = load_stationary_histogram_data(data_path)
            rows.append(
                (
                    stats["lam"],
                    stats["mean"],
                    stats["variance"],
                    stats["std"],
                )
            )
        except (ValueError, OSError):
            continue

    rows.sort(key=lambda r: r[0])

    header = "{:<8} {:>14} {:>14} {:>14}".format(
        "lambda", "mean", "variance", "std_dev"
    )
    print(header)
    print("-" * len(header))

    with open(stats_path, "w", encoding="utf-8") as f:
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")

        for lam, mean, variance, std_dev in rows:
            line = "{:<8.2f} {:>14.8e} {:>14.8e} {:>14.8e}".format(
                lam, mean, variance, std_dev
            )
            print(line)
            f.write(line + "\n")


def write_all_stationary_histograms(
    lambdas: np.ndarray | None = None,
    n: int = 2000,
    eps: float = 1e-10,
    output_dir: str | None = None,
) -> None:
    """
    Compute stationary distributions for a grid of lambdas and write histogram
    data to .txt files under outputs/model1/stationary by default.
    """
    lambda_values = np.atleast_1d(
        np.linspace(0.00, 0.99, 100) if lambdas is None else lambdas
    )
    target_dir = output_dir or os.path.join(BASE_OUTPUT_DIR, f"{n}nodes", "stationary")
    os.makedirs(target_dir, exist_ok=True)

    for lam in lambda_values:
        lam_f = float(lam)
        pi, bins = simulate(lam_f, n, eps)
        mean, variance, std_dev = distribution_stats(pi, bins)
        data_path = os.path.join(
            target_dir, f"model1_stationary_hist_lambda_{lam_f:.2f}.txt"
        )
        with open(data_path, "w", encoding="utf-8") as f:
            f.write("n lam mean variance std\n")
            f.write(
                f"{n} {lam_f:.6f} {mean:.6f} {variance:.6f} {std_dev:.6f}\n"
            )
            f.write("bin_midpoint probability\n")
            centers = 0.5 * (bins[:-1] + bins[1:])
            for mid, prob in zip(centers, pi):
                f.write(f"{mid:.6f} {prob:.6f}\n")


def load_stationary_histogram_data(
    histogram_data_path: str,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    """Load saved stats and bin midpoint/probability data from a histogram file."""
    with open(histogram_data_path, "r", encoding="utf-8") as f:
        stats_header = f.readline().split()
        stats_values = f.readline().split()
        hist_header = f.readline().split()

        if stats_header != ["n", "lam", "mean", "variance", "std"]:
            raise ValueError(
                f"Unexpected stats header in {histogram_data_path}: {stats_header}"
            )
        if hist_header != ["bin_midpoint", "probability"]:
            raise ValueError(
                f"Unexpected histogram header in {histogram_data_path}: {hist_header}"
            )
        if len(stats_values) != 5:
            raise ValueError(
                f"Malformed stats row in {histogram_data_path}."
            )

        stats = {
            "n": int(stats_values[0]),
            "lam": float(stats_values[1]),
            "mean": float(stats_values[2]),
            "variance": float(stats_values[3]),
            "std": float(stats_values[4]),
        }

        midpoints = []
        probabilities = []
        for line in f:
            parts = line.split()
            if not parts:
                continue
            midpoints.append(float(parts[0]))
            probabilities.append(float(parts[1]))

    if not midpoints:
        raise ValueError(
            f"No histogram rows were found in {histogram_data_path}."
        )

    return stats, np.array(midpoints), np.array(probabilities)


def plot_histogram_from_data(
    histogram_data_path: str,
    plot_path: str,
    title: str | None = None,
) -> None:
    """Draw one histogram from a saved histogram-data file."""
    stats, midpoints, probabilities = load_stationary_histogram_data(
        histogram_data_path
    )
    # Reconstruct bins from midpoints: I_j = (j - 1/2, j + 1/2], so edges at 0.5, 1.5, ..., n+0.5
    bins = np.concatenate([[midpoints[0] - 0.5], midpoints + 0.5])
    if title is None:
        title = (
            f"Model One stationary distribution (lambda={stats['lam']:.2f})"
        )
    draw_histogram(
        probabilities,
        bins,
        output_path=plot_path,
        title=title,
    )


def second_largest_eigenvalue(P: np.ndarray) -> float:
    """
    Return the second-largest eigenvalue modulus of the transition matrix P.

    For an irreducible, aperiodic Markov chain, the largest eigenvalue is 1.
    The second-largest eigenvalue (in absolute value) controls the spectral gap.
    """
    eigvals = np.linalg.eigvals(P.T)
    moduli = np.sort(np.abs(eigvals))[::-1]
    if moduli.size < 2:
        return 0.0
    return float(moduli[1])


def spectral_mixing_time(eps: float, lambda2: float) -> float:
    """
    Spectral proxy for the ε-mixing time based on the spectral gap 1 - lambda2.
    """
    if not (0.0 < eps < 1.0):
        raise ValueError("eps must be in (0, 1).")
    if lambda2 >= 1.0:
        return float("inf")
    gap = 1.0 - lambda2
    return float(np.log(1.0 / eps) / gap)


def write_mixing_time_spectral(
    n: int = 2000,
    eps_stationary: float = 1e-10,
    eps_mixing: float = 1e-3,
    output_dir: str | None = None,
) -> None:
    """
    For a grid of lambdas, compute a spectral mixing-time proxy and write
    results under outputs/model1/{n}nodes/mixing by default.

    The proxy is
        t_mix^(spec)(eps_mixing, lambda) ≈ log(1/eps_mixing) / (1 - lambda2(lambda)),
    where lambda2(lambda) is the second-largest eigenvalue modulus of P(lambda).
    """
    mixing_dir = output_dir or os.path.join(BASE_OUTPUT_DIR, f"{n}nodes", "mixing")
    os.makedirs(mixing_dir, exist_ok=True)
    out_path = os.path.join(mixing_dir, "model1_mixing_spectral.txt")
    lambdas = np.linspace(0.0, 0.99, 100)

    header = "{:<8} {:>14} {:>14} {:>14}".format(
        "lambda", "lambda2", "spectral_gap", "t_mix_spec"
    )
    print(header)
    print("-" * len(header))

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")

        for lam in lambdas:
            P, _ = build_transition_matrix(lam, n, renormalize=True)
            lambda2 = second_largest_eigenvalue(P)
            spectral_gap = 1.0 - lambda2
            t_mix_spec = spectral_mixing_time(eps_mixing, lambda2)

            line = "{:<8.2f} {:>14.8e} {:>14.8e} {:>14.8e}".format(
                lam, lambda2, spectral_gap, t_mix_spec
            )
            print(line)
            f.write(line + "\n")


def generate_stationary_histograms(
    lambdas: np.ndarray | None = None,
    n: int = 2000,
    eps: float = 1e-10,
    output_dir: str | None = None,
) -> None:
    write_all_stationary_histograms(lambdas=lambdas, n=n, eps=eps, output_dir=output_dir)


def generate_stationary_stats(data_dir: str | None = None) -> None:
    write_all_stats_to_file(data_dir=data_dir)


def generate_mixing_time_spectral(
    n: int = 2000,
    eps_mixing: float = 1e-3,
    output_dir: str | None = None,
) -> None:
    write_mixing_time_spectral(
        n=n, eps_stationary=1e-10, eps_mixing=eps_mixing, output_dir=output_dir
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Model One: stationary distribution and mixing time."
    )
    parser.add_argument(
        "n",
        type=int,
        nargs="?",
        default=1000,
        help="Number of nodes (default: 1000)",
    )
    args = parser.parse_args()
    n = args.n

    eps_stationary = 1e-10
    eps_mixing = 1e-3

    run_stationary_histograms = True
    run_stationary_stats = True
    run_mixing_spectral = True

    nodes_dir = os.path.join(BASE_OUTPUT_DIR, f"{n}nodes")
    stationary_dir = os.path.join(nodes_dir, "stationary")
    mixing_dir = os.path.join(nodes_dir, "mixing")

    if run_stationary_histograms:
        generate_stationary_histograms(n=n, eps=eps_stationary, output_dir=stationary_dir)
        for fname in sorted(os.listdir(stationary_dir)):
            if not fname.endswith(".txt"):
                continue
            data_path = os.path.join(stationary_dir, fname)
            plot_path = os.path.join(
                stationary_dir, f"{os.path.splitext(fname)[0]}.png"
            )
            plot_histogram_from_data(data_path, plot_path)
    if run_stationary_stats:
        generate_stationary_stats(data_dir=stationary_dir)
    if run_mixing_spectral:
        generate_mixing_time_spectral(n, eps_mixing, output_dir=mixing_dir)
