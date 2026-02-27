import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import math
import time

def median_of_three(a, b, c):
    """Return the median of three values"""
    return sorted([a, b, c])[1]

def center_of_two(a, b):
    """Return the center of two values"""
    return (a + b) / 2

def simulate_adaptive_stances(n, T, seed, ll=1):
    """
    Simulate the opinion formation model where agents update their stances.

    Args:
        n: number of agents (and alternatives)
        T: number of deliberation rounds
        seed: random seed for reproducibility

    Returns:
        final outcome o_T
    """
    rng = np.random.RandomState(seed)
    rng_stance = np.random.RandomState(seed + 1000000)  # Different RNG for stance updates

    # Initialize
    bliss_points = {u: u for u in range(1, n+1)}  # a*_u = u for u in [n]
    stances = {u: u for u in range(1, n+1)}  # a^0_u = a*_u
    o_prev = rng.randint(1, n+1)  # o^0 is random position

    # mark, check if it the first time a node is shown
    mark_first_time = {u: True for u in range(1, n+1)}

    # Deliberation rounds
    for t in range(1, T+1):
        # Randomly select two agents
        i, j = rng.choice(range(1, n+1), size=2, replace=True) # sj: should be with replacement

        stances[i] = bliss_points[i]*(1 - ll) + o_prev * ll
        stances[j] = bliss_points[j]*(1 - ll) + o_prev * ll
        o_current = median_of_three(stances[i], stances[j], o_prev)

        # if mark_first_time[i]:
        #     stances[i] = center_of_two(bliss_points[i], o_prev) 
        #     mark_first_time[i] = False
        # else:
        #     stances[i] = median_of_three(bliss_points[i], stances[i], o_prev)
        
        # if mark_first_time[j]:
        #     stances[j] = center_of_two(bliss_points[j], o_prev)
        #     mark_first_time[j] = False
        # else:
        #     stances[j] = median_of_three(bliss_points[j], stances[j], o_prev)

        # Current outcome: median of two stances and previous outcome

        # Update stances for agents i and j
        # Agent i
        # if stances[i] == bliss_points[i]:
        #     # With prob 2/3 stay at bliss point, with prob 1/3 move to outcome
        #     if rng_stance.random() < 2/3:
        #         stances[i] = bliss_points[i]
        #     else:
        #         stances[i] = o_current
        # else:
        # stances[i] = median_of_three(bliss_points[i], stances[i], o_current)

        # Agent j
        # if stances[j] == bliss_points[j]:
        #     # With prob 2/3 stay at bliss point, with prob 1/3 move to outcome
        #     if rng_stance.random() < 2/3:
        #         stances[j] = bliss_points[j]
        #     else:
        #         stances[j] = o_current
        # else:
        # stances[j] = median_of_three(bliss_points[j], stances[j], o_current)

        # Update outcome
        o_prev = o_current

    return o_prev

def simulate_fixed_stances(n, T, seed):
    """
    Simulate the opinion formation model where agents always use their bliss points.

    Args:
        n: number of agents (and alternatives)
        T: number of deliberation rounds
        seed: random seed for reproducibility

    Returns:
        final outcome o_T
    """
    rng = np.random.RandomState(seed)

    # Initialize
    bliss_points = {u: u for u in range(1, n+1)}  # a*_u = u for u in [n]
    o_prev = rng.randint(1, n+1)  # o^0 is random position

    # Deliberation rounds
    for t in range(1, T+1):
        # Randomly select two agents
        i, j = rng.choice(range(1, n+1), size=2, replace=True) # sj: should be with replacement

        # Current outcome: median of two bliss points and previous outcome
        o_current = median_of_three(bliss_points[i], bliss_points[j], o_prev)

        # Update outcome
        o_prev = o_current

    return o_prev

def compare_models(n, K, repeat = 50, ll=0.5):
    """
    Run K simulations of both models, print results, plot distributions, and output variance.

    Args:
        n: number of agents
        K: number of simulation runs
    """
    T_base = 100
    # repeat = 50
    print(f"Running simulations with n={n}, K={K}, T_base={T_base}")
    print("="*60)

    for x in range(repeat):
        adaptive_outcomes = []
        fixed_outcomes = []
        T = T_base * (x+1)
        print(f"T: {T}")

        print(f"Running simulation {x+1} of {repeat} with ll={ll}")
        # Run simulations for both models with time-based seeds
        for k in range(K):
            adaptive_outcomes.append(simulate_adaptive_stances(n, T, seed=k, ll=ll))
            fixed_outcomes.append(simulate_fixed_stances(n, T, seed=k))

        # Calculate distributions
        adaptive_dist = Counter(adaptive_outcomes)
        fixed_dist = Counter(fixed_outcomes)

        # print(f"Adaptive dist: {adaptive_dist}")
        # print(f"Fixed dist: {fixed_dist}")

        # Calculate statistics
        adaptive_mean = np.mean(adaptive_outcomes)
        fixed_mean = np.mean(fixed_outcomes)
        adaptive_var = np.var(adaptive_outcomes)
        fixed_var = np.var(fixed_outcomes)

        # Print distributions
        print("\nAdaptive Stances Distribution:")
        for outcome in sorted(adaptive_dist.keys()):
            print(f"  Outcome {outcome}: {adaptive_dist[outcome]} ({100*adaptive_dist[outcome]/K:.2f}%)")

        print("\nFixed Stances Distribution:")
        for outcome in sorted(fixed_dist.keys()):
            print(f"  Outcome {outcome}: {fixed_dist[outcome]} ({100*fixed_dist[outcome]/K:.2f}%)")

        # Print summary statistics
        print("\n" + "="*60)
        print("Summary Statistics:")
        print(f"  Median position: {(n+1)/2:.1f}")
        print(f"\nAdaptive Stances:")
        print(f"  Mean: {adaptive_mean:.4f}")
        print(f"  Variance: {adaptive_var:.4f}")
        print(f"  Std Dev: {np.sqrt(adaptive_var):.4f}")
        print(f"\nFixed Stances:")
        print(f"  Mean: {fixed_mean:.4f}")
        print(f"  Variance: {fixed_var:.4f}")
        print(f"  Std Dev: {np.sqrt(fixed_var):.4f}")
        print("="*60)

        # Plot distributions
        # positions = range(1, n+1)
        adaptive_probs = [adaptive_dist.get(i, 0) / K for i in sorted(adaptive_dist.keys())]
        # print(f"Adaptive probs: {adaptive_probs}")
        # print(f"sorted adaptive keys: {sorted(adaptive_dist.keys())}")
        fixed_probs = [fixed_dist.get(i, 0) / K for i in sorted(fixed_dist.keys())]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Adaptive stances
        ax1.hist(sorted(adaptive_dist.keys()), bins=len(adaptive_probs), weights=adaptive_probs, alpha=0.7, color='blue')
        ax1.axvline((n+1)/2, color='red', linestyle='--', linewidth=2, label='Median position')
        ax1.axvline(adaptive_mean, color='green', linestyle='--', linewidth=2, label=f'Mean={adaptive_mean:.2f}')
        ax1.set_xlabel('Final Outcome Position')
        ax1.set_ylabel('Probability')
        ax1.set_title(f'Adaptive Stances Model (Var={adaptive_var:.2f}, STD dev={np.sqrt(adaptive_var):.2f})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Fixed stances
        ax2.bar(sorted(fixed_dist.keys()), fixed_probs, alpha=0.7, color='red')
        ax2.axvline((n+1)/2, color='red', linestyle='--', linewidth=2, label='Median position')
        ax2.axvline(fixed_mean, color='green', linestyle='--', linewidth=2, label=f'Mean={fixed_mean:.2f}')
        ax2.set_xlabel('Final Outcome Position')
        ax2.set_ylabel('Probability')
        ax2.set_title(f'Fixed Stances Model (Var={fixed_var:.2f}, STD dev={np.sqrt(fixed_var):.2f})')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'opinion_formation_distributions_{x+1}.png', dpi=150, bbox_inches='tight')
        plt.close()
        # plt.show()

        # return {
        #     'adaptive_mean': adaptive_mean,
        #     'adaptive_var': adaptive_var,
        #     'fixed_mean': fixed_mean,
        #     'fixed_var': fixed_var,
        #     'adaptive_dist': adaptive_dist,
        #     'fixed_dist': fixed_dist
        # }

# Example usage
if __name__ == "__main__":
    n = 10
    repeat = 10
    K = 1000
    lls = 0.5

    results = compare_models(n, K, repeat, ll=0.5)