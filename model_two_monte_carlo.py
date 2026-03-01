"""
Model Two Monte Carlo simulation utilities.

This module contains the simulation core for Model Two:
agents deliberate first (Nash bargaining = median of three in the median graph
setting), then the two chosen agents update their stances by convex combination
with the current outcome. Follows the median-of-three convention from Fain et al.
on a line graph.

- Per round t: (a) uniformly sample two agents i, j; stances at start of round
  are a_i^{t-1}, a_j^{t-1}. (b) Outcome o^t = Median(a_i^{t-1}, a_j^{t-1}, o^{t-1}).
  (c) Agents i and j update: a_i^t = a_i^*(1 - λ) + o^t λ, a_j^t = a_j^*(1 - λ) + o^t λ,
  with λ ∈ (0, 1). (d) All other agents u: a_u^t = a_u^{t-1}.
- Discrete positions in {1, ..., n}, bliss points a*_u = u, initial stances a^0_u = u.

Lambda-sweep generation, summary-file reporting, and sweep plots live in
`model_two_generate_lambda_histograms.py` and
`model_two_varaince_mixing_plot.py`.
"""

import os
import numpy as np
from collections import Counter


def _median_of_three(a, b, c):
    """Return the median of three values (Nash bargaining in median graph setting)."""
    return sorted([a, b, c])[1]


def simulate_model_two(n_agents, n_rounds, seed, lam=0.5, record_at=None):
    """
    Simulate one run of Model Two: deliberate then update stances.

    Args:
        n_agents: number of agents (and alternatives), positions in [n].
        n_rounds: number of deliberation rounds T.
        seed: random seed for reproducibility.
        lam: λ in (0, 1); stance update a^t = a^*(1 - λ) + o^t λ.
        record_at: optional set/list of round indices at which to record the outcome;
            if given, returns (final_outcome, trajectory) with trajectory[t] = outcome at round t.

    Returns:
        If record_at is None: final outcome o_T (int in [1, n_agents]).
        Else: (final_outcome, trajectory_dict) with trajectory_dict[t] = outcome at round t for t in record_at.
    """
    rng = np.random.RandomState(seed)
    # Bliss points a*_u = u; initial stances a^0_u = u
    bliss = {u: u for u in range(1, n_agents + 1)}
    stance = {u: u for u in range(1, n_agents + 1)}
    outcome_prev = rng.randint(1, n_agents + 1)

    trajectory = {} if record_at is not None else None
    record_set = set(record_at) if record_at is not None else set()

    for round_t in range(1, n_rounds + 1):
        i, j = rng.choice(range(1, n_agents + 1), size=2, replace=True)
        outcome_curr = _median_of_three(stance[i], stance[j], outcome_prev)
        # a_i^t = a_i^*(1 - λ) + o^t λ, then round and clamp to [1, n]
        val_i = bliss[i] * (1 - lam) + outcome_curr * lam
        stance[i] = max(1, min(n_agents, int(round(val_i))))
        val_j = bliss[j] * (1 - lam) + outcome_curr * lam
        stance[j] = max(1, min(n_agents, int(round(val_j))))
        outcome_prev = outcome_curr
        if round_t in record_set:
            trajectory[round_t] = outcome_prev

    if record_at is None:
        return outcome_prev
    return outcome_prev, trajectory


def _empirical_distribution(outcomes, n_agents):
    """Outcomes: list of ints in [1, n_agents]. Return probability vector over 1..n_agents."""
    counts = np.zeros(n_agents + 1, dtype=float)  # index 0 unused
    for o in outcomes:
        counts[o] += 1
    counts = counts[1:]
    total = counts.sum()
    if total == 0:
        return np.ones(n_agents) / n_agents
    return counts / total


def _total_variation(pi, pi_ref):
    """TV distance between two probability vectors (same length)."""
    return 0.5 * np.abs(pi - pi_ref).sum()


def _write_histogram_data(
    path,
    dist,
    n_runs,
    n_agents,
    n_rounds,
    lam,
    mean_outcome,
    var_outcome,
    std_outcome,
):
    """Write summary stats and per-outcome counts needed to draw a histogram later."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("n_agents n_rounds n_runs lam mean variance std\n")
        f.write(
            f"{n_agents} {n_rounds} {n_runs} {lam} "
            f"{mean_outcome:.6f} {var_outcome:.6f} {std_outcome:.6f}\n"
        )
        f.write("outcome count fraction\n")
        for outcome in range(1, n_agents + 1):
            count = dist.get(outcome, 0)
            f.write(f"{outcome} {count} {count / n_runs:.6f}\n")


def estimate_empirical_mixing_time(
    n_agents,
    T_ref,
    n_chains,
    time_grid=None,
    eps=0.1,
    lam=0.5,
    seed=0,
):
    """
    Estimate empirical ε-mixing time for the simulation core.

    Run n_chains independent chains for T_ref rounds. At each checkpoint in
    time_grid, compute the empirical distribution of the outcome over chains and
    compare it to the empirical distribution at T_ref. Return the smallest
    checkpoint such that TV <= eps.

    Args:
        n_agents: number of agents/positions.
        T_ref: reference horizon (chains run for this many rounds).
        n_chains: number of independent chains.
        time_grid: list of round indices at which to compare; if None, use a default grid.
        eps: mixing tolerance (TV distance threshold).
        lam: λ in (0, 1) for stance updates.
        seed: base random seed (chain k uses seed + k).

    Returns:
        t_mix: estimated mixing time (smallest t in time_grid with TV(π_t, π_T_ref) <= eps,
            or T_ref if none).
        tv_curve: dict mapping t -> TV(π_t, π_T_ref) for t in time_grid (for diagnostics).
    """
    checkpoints = sorted(set(time_grid or [10, 50, 100, 200, 500, 1000, 2000, 5000]) | {T_ref})

    # outcomes_at_t[t] = list of outcomes at time t across chains
    outcomes_at_t = {t: [] for t in checkpoints}
    for k in range(n_chains):
        _, traj = simulate_model_two(
            n_agents, T_ref, seed=seed + k, lam=lam, record_at=checkpoints
        )
        for t, o in traj.items():
            outcomes_at_t[t].append(o)

    pi_ref = _empirical_distribution(outcomes_at_t[T_ref], n_agents)
    tv_curve = {}
    t_mix = T_ref
    for t in checkpoints:
        if t > T_ref:
            continue
        pi_t = _empirical_distribution(outcomes_at_t[t], n_agents)
        tv = _total_variation(pi_t, pi_ref)
        tv_curve[t] = tv
        if tv <= eps and t < t_mix:
            t_mix = t
    return t_mix, tv_curve


def run_model_two_simulation(
    n_agents,
    n_runs,
    n_rounds=None,
    T_ref=5000,
    n_chains_mixing=1000,
    mixing_multiplier=2,
    T_min=500,
    eps_mixing=0.1,
    lam=0.5,
    seed=0,
    output_dir=None,
    histogram_path=None,
):
    """
    Run the Model Two Monte Carlo simulation for one lambda value.

    If n_rounds is None, estimate empirical t_mix and set n_rounds = max(T_min, mixing_multiplier * t_mix).
    Then run n_runs independent chains for n_rounds rounds, collect final
    outcomes, compute distribution and variance, and optionally save
    histogram-ready data under output_dir. Multi-lambda analysis and summary
    plotting are handled in the separate analysis module.

    Args:
        n_agents: number of agents.
        n_runs: number of chains for the main simulation.
        n_rounds: if None, set from empirical mixing time; else use this value.
        T_ref: reference horizon for mixing-time estimation.
        n_chains_mixing: number of chains used to estimate t_mix.
        mixing_multiplier: n_rounds >= mixing_multiplier * t_mix when n_rounds is None.
        T_min: minimum n_rounds when n_rounds is None.
        eps_mixing: TV threshold for empirical mixing time.
        lam: λ in (0, 1) for stance updates.
        seed: base random seed.
        output_dir: if set, save model2_empirical_mixing.txt and histogram-ready
            distribution data under this directory.
        histogram_path: if set, save histogram-ready distribution data to this path
            instead of output_dir/model2_histogram_data.txt.

    Returns:
        dict with keys: n_rounds, t_mix (if estimated), final_outcomes, distribution (Counter),
        mean, variance, std.
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), "outputs", "model2")
    os.makedirs(output_dir, exist_ok=True)

    t_mix_est = None
    if n_rounds is None:
        t_mix_est, tv_curve = estimate_empirical_mixing_time(
            n_agents, T_ref, n_chains_mixing, eps=eps_mixing, lam=lam, seed=seed
        )
        n_rounds = max(T_min, int(mixing_multiplier * t_mix_est))
        # Save mixing diagnostics
        mix_path = os.path.join(output_dir, "model2_empirical_mixing.txt")
        with open(mix_path, "w", encoding="utf-8") as f:
            f.write("n_agents T_ref n_chains eps lam t_mix n_rounds_used\n")
            f.write(
                f"{n_agents} {T_ref} {n_chains_mixing} {eps_mixing} {lam} {t_mix_est} {n_rounds}\n"
            )
            f.write("t TV\n")
            for t in sorted(tv_curve.keys()):
                f.write(f"{t} {tv_curve[t]:.6e}\n")
        print(f"Empirical t_mix(eps={eps_mixing}) = {t_mix_est}, using n_rounds = {n_rounds}")

    final_outcomes = []
    for k in range(n_runs):
        o_T = simulate_model_two(n_agents, n_rounds, seed=seed + k, lam=lam)
        final_outcomes.append(o_T)

    dist = Counter(final_outcomes)
    mean_outcome = np.mean(final_outcomes)
    var_outcome = np.var(final_outcomes)
    std_outcome = np.sqrt(var_outcome)

    hist_path = (
        histogram_path
        if histogram_path is not None
        else os.path.join(output_dir, "model2_histogram_data.txt")
    )
    _write_histogram_data(
        hist_path,
        dist,
        n_runs,
        n_agents,
        n_rounds,
        lam,
        mean_outcome,
        var_outcome,
        std_outcome,
    )

    result = {
        "n_rounds": n_rounds,
        "t_mix": t_mix_est,
        "final_outcomes": final_outcomes,
        "distribution": dist,
        "mean": mean_outcome,
        "variance": var_outcome,
        "std": std_outcome,
    }
    return result
