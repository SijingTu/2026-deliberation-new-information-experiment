"""
Model One: discretized Markov chain on [0, 1] and its stationary distribution.

We follow the model one in the note. There is a continuum of agents,
each with a fixed bliss point a*_u in [0, 1], drawn i.i.d. from the uniform
distribution. At deliberation round t, the state X_t is the current alternative
o_t in [0, 1].

At each round:
- Sample two agents u and v uniformly at random.
- Given the previous outcome X_{t-1}, form temporary stances
    a_u^t = (1 - λ) a*_u + λ X_{t-1},
    a_v^t = (1 - λ) a*_v + λ X_{t-1},
  where λ in (0, 1) controls how responsive stances are to the previous outcome.
- The new outcome is the median of the two stances and the previous outcome:
    X_t = median(a_u^t, a_v^t, X_{t-1}).

Under the assumptions (a*_u i.i.d. Uniform[0, 1] and infinitely many agents),
this defines a one-dimensional Markov chain (X_t) with an explicit conditional
CDF K(x, z) = P(X_t <= z | X_{t-1} = x). The function build_transition_matrix
implements a standard discretization of this kernel on a uniform grid:

- Partition [0, 1] into M bins with edges b[0], ..., b[M] and midpoints
  x[0], ..., x[M-1].
- Interpret state index k as the midpoint x[k] of bin k.
- Use the CDF K(x[k], z) to obtain the probability that the next state falls
  into bin m via
      P[k, m] = K(x[k], b[m+1]) - K(x[k], b[m]).

The resulting row-stochastic matrix P is an approximation to the transition
kernel of Model One restricted to this grid. The stationary distribution pi is
then approximated numerically by power iteration on P.
"""

import os
import numpy as np
import matplotlib.pyplot as plt


BASE_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs", "model1")
STATIONARY_DIR = os.path.join(BASE_OUTPUT_DIR, "stationary")
MIXING_DIR = os.path.join(BASE_OUTPUT_DIR, "mixing")

def build_transition_matrix(lam: float, M: int, renormalize: bool = True):
    """
    Build the MxM transition matrix P for the discretized Model One kernel K(x, z).

    Each row k corresponds to the current state X_{t-1} = x[k] (the midpoint of
    bin k). The (k, m) entry is the probability that X_t falls into bin m,
    computed by integrating the conditional CDF K(x[k], z) over bin m.
    """
    if not (0 <= lam < 1):
        raise ValueError("lambda must be in [0, 1).")

    # bins b[0..M] and midpoints x[1..M]
    b = np.linspace(0.0, 1.0, M + 1)
    x = 0.5 * (b[:-1] + b[1:])  # length M

    def K(x_val, z_val):
        L = lam * x_val
        U = lam * x_val + 1.0 - lam
        if z_val < L:
            return 0.0
        if z_val > U:
            return 1.0
        h = (z_val - L) / (1.0 - lam)
        if z_val < x_val:
            return h * h
        else:
            return 1.0 - (1.0 - h) * (1.0 - h)

    # Build P[k, m]
    P = np.zeros((M, M), dtype=float)
    for k in range(M):
        for m in range(M):
            P[k, m] = K(x[k], b[m + 1]) - K(x[k], b[m])

        if renormalize:
            row_sum = P[k].sum()
            if row_sum > 0:
                P[k] /= row_sum

    return P, b

def power_iteration_stationary(P: np.ndarray, eps: float = 1e-10, max_iter: int = 1_000_000):
    """
    Power iteration on row-stochastic P to find stationary distribution pi.
    """
    M = P.shape[0]
    pi = np.ones(M) / M

    for _ in range(max_iter):
        pi_new = pi @ P
        if np.linalg.norm(pi_new - pi, ord=1) < eps:
            return pi_new
        pi = pi_new

    return pi  # return last iterate if not converged

def simulate(lam: float, M: int = 2000, eps: float = 1e-10, renormalize: bool = True):
    P, bins = build_transition_matrix(lam, M, renormalize=renormalize)
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

def write_all_stats_to_file(M: int = 2000, eps: float = 1e-10) -> None:
    """
    Compute basic statistics of the stationary distribution for a grid of lambdas
    and write them to a table under outputs/model1/stationary.
    """
    lambdas = np.linspace(0.0, 0.99, 100)
    os.makedirs(STATIONARY_DIR, exist_ok=True)
    stats_path = os.path.join(STATIONARY_DIR, "model1_stationary_stats.txt")

    header = "{:<8} {:>14} {:>14} {:>14}".format(
        "lambda", "mean", "variance", "std_dev"
    )
    print(header)
    print("-" * len(header))

    with open(stats_path, "w", encoding="utf-8") as f:
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")

        for lam in lambdas:
            pi, bins = simulate(lam, M, eps)
            mean, variance, std_dev = distribution_stats(pi, bins)
            line = "{:<8.2f} {:>14.8e} {:>14.8e} {:>14.8e}".format(
                lam, mean, variance, std_dev
            )
            print(line)
            f.write(line + "\n")


def write_all_stationary_histograms(
    M: int = 2000,
    eps: float = 1e-10,
    output_dir: str | None = None,
) -> None:
    """
    Generate stationary-distribution histograms for a grid of lambdas and save
    them under outputs/model1/stationary by default.
    """
    lambdas = np.linspace(0.00, 0.99, 100)
    target_dir = output_dir or STATIONARY_DIR
    os.makedirs(target_dir, exist_ok=True)

    for lam in lambdas:
        pi, bins = simulate(lam, M, eps)
        output_path = os.path.join(
            target_dir, f"model1_stationary_hist_lambda_{lam:.2f}.png"
        )
        title = f"Model One stationary distribution (lambda={lam:.2f})"
        draw_histogram(pi, bins, output_path=output_path, title=title)


def second_largest_eigenvalue(P: np.ndarray) -> float:
    """
    Return the second-largest eigenvalue modulus of the transition matrix P.

    For an irreducible, aperiodic Markov chain, the largest eigenvalue is 1.
    The second-largest eigenvalue (in absolute value) controls the spectral gap.
    """
    # Use eigenvalues of P^T; eigenvalues are the same as for P.
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
    M: int = 2000,
    eps_stationary: float = 1e-10,
    eps_mixing: float = 1e-3,
) -> None:
    """
    For a grid of lambdas, compute a spectral mixing-time proxy and write
    results under outputs/model1/mixing.

    The proxy is
        t_mix^(spec)(eps_mixing, lambda) ≈ log(1/eps_mixing) / (1 - lambda2(lambda)),
    where lambda2(lambda) is the second-largest eigenvalue modulus of P(lambda).
    """
    os.makedirs(MIXING_DIR, exist_ok=True)
    out_path = os.path.join(MIXING_DIR, "model1_mixing_spectral.txt")
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
            P, _ = build_transition_matrix(lam, M, renormalize=True)
            lambda2 = second_largest_eigenvalue(P)
            spectral_gap = 1.0 - lambda2
            t_mix_spec = spectral_mixing_time(eps_mixing, lambda2)

            line = "{:<8.2f} {:>14.8e} {:>14.8e} {:>14.8e}".format(
                lam, lambda2, spectral_gap, t_mix_spec
            )
            print(line)
            f.write(line + "\n")


def generate_stationary_histograms(M: int = 2000, eps: float = 1e-10) -> None:
    write_all_stationary_histograms(M, eps)


def generate_stationary_stats(M: int = 2000, eps: float = 1e-10) -> None:
    write_all_stats_to_file(M, eps)


def generate_mixing_time_spectral(M: int = 2000, eps_mixing: float = 1e-3) -> None:
    write_mixing_time_spectral(M=M, eps_stationary=1e-10, eps_mixing=eps_mixing)


if __name__ == "__main__":
    M = 2000
    eps_stationary = 1e-10
    eps_mixing = 1e-3

    run_stationary_histograms = True
    run_stationary_stats = True
    run_mixing_spectral = True

    if run_stationary_histograms:
        generate_stationary_histograms(M, eps_stationary)
    if run_stationary_stats:
        generate_stationary_stats(M, eps_stationary)
    if run_mixing_spectral:
        generate_mixing_time_spectral(M, eps_mixing)
