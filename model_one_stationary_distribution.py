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
    lambdas = np.linspace(0.0, 0.99, 100)

    header = "{:<8} {:>14} {:>14} {:>14}".format(
        "lambda", "mean", "variance", "std_dev"
    )
    print(header)
    print("-" * len(header))

    for lam in lambdas:
        pi, bins = simulate(lam, M, eps)
        mean, variance, std_dev = distribution_stats(pi, bins)
        print("{:<8.2f} {:>14.8e} {:>14.8e} {:>14.8e}".format(
            lam, mean, variance, std_dev
        ))
        with open("stats.txt", "a") as f:
            f.write("{:<8.2f} {:>14.8e} {:>14.8e} {:>14.8e}\n".format(
                lam, mean, variance, std_dev
            ))


def write_all_histograms(M: int = 2000, eps: float = 1e-10, output_dir: str = "sim") -> None:
    lambdas = np.linspace(0.00, 0.99, 100)
    target_dir = os.path.join(os.path.dirname(__file__), output_dir)
    os.makedirs(target_dir, exist_ok=True)

    for lam in lambdas:
        pi, bins = simulate(lam, M, eps)
        output_path = os.path.join(target_dir, f"hist_lambda_{lam:.2f}.png")
        title = f"Stationary distribution (lambda={lam:.2f})"
        draw_histogram(pi, bins, output_path=output_path, title=title)


if __name__ == "__main__":
    # lam = 0.5
    M = 2000
    eps = 1e-10
    # pi, bins = simulate(lam, M, eps)
    # print("pi sum:", pi.sum())
    # print("bins shape:", bins.shape)
    # print("pi:", pi)
    # print("bins:", bins)

    # mean, variance, std_dev = distribution_stats(pi, bins)
    # print("mean:", mean)
    # print("variance:", variance)
    # print("std_dev:", std_dev)

    # write_all_stats_to_file(M, eps)
    write_all_histograms(M, eps, output_dir="sim")
