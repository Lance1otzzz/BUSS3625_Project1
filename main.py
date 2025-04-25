# File: main.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time # For timing runs

from bandit import KArmedBandit
from experiment import BanditExperiment
from strategies import CombinedStrategies, CommitmentStrategies # Dynamic strategies currently not integrated
# Import updated visualization functions
from visualization import (
    visualize_single_run_details,
    visualize_strategy_comparison,
    visualize_average_regret_over_time,
    visualize_optimal_commitment_probability
)
from optimization import objective_factory, run_optuna_optimization

# --- Helper Function to Calculate Step-by-Step Regret ---
def calculate_cumulative_regret_per_step(experiment: BanditExperiment) -> np.ndarray:
    """
    Calculates the cumulative regret at each step of a completed experiment.

    Args:
        experiment: A BanditExperiment object after T+N rounds have run.

    Returns:
        A numpy array of shape (T+N,) containing cumulative regret at each step.
    """
    T = experiment.T
    N = experiment.N
    total_rounds = T + N
    if total_rounds == 0:
        return np.array([])

    optimal_mean = experiment.bandit.get_optimal_mean()
    true_means = experiment.bandit.get_true_means()

    # Combine arms chosen and rewards from both phases
    arms_chosen = experiment.experiment_arms_chosen
    rewards_received = experiment.experiment_rewards_received
    if N > 0 and experiment.commitment_arm_selected is not None:
        commitment_arm = experiment.commitment_arm_selected
        # Pad arms_chosen with the commitment arm for N rounds
        arms_chosen.extend([commitment_arm] * N)
        # Rewards received already includes both phases if run correctly
        rewards_received.extend(experiment.commitment_rewards_received)


    cumulative_regret = np.zeros(total_rounds)
    actual_cumulative_reward = 0
    optimal_cumulative_reward = 0

    for t in range(total_rounds):
        if t < len(arms_chosen): # Check bounds
            chosen_arm = arms_chosen[t]
            reward = rewards_received[t] # Use actual received reward for total actual reward

            # Expected reward for chosen arm (used for regret calculation standard definition)
            expected_reward_chosen_arm = true_means[chosen_arm]

            # Calculate regret relative to OPTIMAL arm's EXPECTED reward
            # Regret(t) = E[R_opt] - E[R_chosen(t)]
            instantaneous_regret = optimal_mean - expected_reward_chosen_arm

            if t == 0:
                cumulative_regret[t] = instantaneous_regret
            else:
                cumulative_regret[t] = cumulative_regret[t-1] + instantaneous_regret
        # else: # Should not happen if experiment ran fully
        #     if t > 0: cumulative_regret[t] = cumulative_regret[t-1] # Carry over last regret

    # Alternative calculation using total rewards (less common for step-by-step regret plot):
    # cumulative_actual_reward = np.cumsum(rewards_received)
    # cumulative_optimal_reward = optimal_mean * np.arange(1, total_rounds + 1)
    # cumulative_regret_alt = cumulative_optimal_reward - cumulative_actual_reward

    return cumulative_regret


# --- Main Function ---
def main(T=1000, N=500, k=10, num_runs=50, n_trials_optuna=50):
    """
    Main execution function, updated to collect data for new visualizations.

    Args:
        T: Experiment phase rounds.
        N: Commitment phase rounds.
        k: Number of arms.
        num_runs: Number of strategy comparison runs.
        n_trials_optuna: Number of Optuna optimization trials.
    """
    start_time_main = time.time()
    print("="*40)
    print(" K-armed Bandit Experiment - Two-Phase Framework")
    print(f" Parameters: T={T}, N={N}, k={k}, num_runs={num_runs}, n_trials_optuna={n_trials_optuna}")
    print("="*40)

    # --- Define strategy combinations ---
    experiment_policies = {
        f"EpsGreedy(0.1)": lambda exp: CombinedStrategies.epsilon_greedy(exp, epsilon=0.1),
        # f"EpsGreedy(0.01)": lambda exp: CombinedStrategies.epsilon_greedy(exp, epsilon=0.01),
        f"UCB(c=2.0)": lambda exp: CombinedStrategies.ucb(exp, c=2.0),
        # f"UCB(c=1.0)": lambda exp: CombinedStrategies.ucb(exp, c=1.0),
        "ThompsonSampling": CombinedStrategies.thompson_sampling,
        # Add more strategies if desired
    }

    commitment_policies = {
        "Commit:BestEmpirical": CommitmentStrategies.best_empirical,
        # "Commit:MostPulled": CommitmentStrategies.most_pulled,
        # "Commit:LCB(95%)": lambda exp: CommitmentStrategies.confidence_based(exp, confidence_level=0.95)
    }

    # --- 1. Single Run Example ---
    print("\n[Phase 1: Single Run Example]")
    exp_policy_name_single = "UCB(c=2.0)"
    com_policy_name_single = "Commit:BestEmpirical"
    exp_policy_single = experiment_policies[exp_policy_name_single]
    com_policy_single = commitment_policies[com_policy_name_single]

    print(f"Running strategy: {exp_policy_name_single} + {com_policy_name_single}")
    experiment_single = BanditExperiment(k=k, T=T, N=N) # Use fresh bandit instance
    experiment_single.run_experiment_phase(exp_policy_single)
    experiment_single.choose_commitment_arm(com_policy_single)
    experiment_single.run_commitment_phase()
    # Use the updated visualization function
    visualize_single_run_details(
        experiment_single,
        exp_policy_name_single,
        com_policy_name_single,
        filename_prefix="single_run_example"
    )
    print("-" * 30)


    # --- 2. Compare Strategy Combinations (Multiple Runs) ---
    total_combinations = len(experiment_policies) * len(commitment_policies)
    print(f"\n[Phase 2: Comparing {total_combinations} Strategy Combinations ({num_runs} runs each)]")

    # Data storage for results
    final_regret_results = {} # For final avg regret bar chart -> {name: {'mean': m, 'std': s}}
    regret_timeseries_results = {} # For avg regret over time -> {name: [run1_ts, run2_ts, ...]} list of arrays
    optimal_commit_results = {} # For optimal commit prob -> {name: [True, False, ...]} list of bools

    run_counter = 0
    total_runs_to_do = total_combinations * num_runs
    start_time_comparison = time.time()

    for c_name, c_policy in commitment_policies.items():
        for p_name, p_policy in experiment_policies.items():
            run_counter += 1
            full_name = f"{p_name} + {c_name}"
            print(f"  Running Combo {run_counter}/{total_combinations}: {full_name}...")

            # Store results for this combination across runs
            run_regrets = [] # Final regret for each run
            run_regret_timeseries = [] # Regret time series for each run
            run_optimal_commits = [] # Boolean optimal commit for each run

            for run_idx in range(num_runs):
                # Reset experiment environment for each run
                experiment = BanditExperiment(k=k, T=T, N=N)
                try:
                    # Run the two phases
                    experiment.run_experiment_phase(p_policy)
                    experiment.choose_commitment_arm(c_policy)
                    experiment.run_commitment_phase()

                    # Calculate and store final regret
                    final_regret = experiment.calculate_regret()
                    run_regrets.append(final_regret)

                    # Calculate and store step-by-step regret
                    regret_ts = calculate_cumulative_regret_per_step(experiment)
                    run_regret_timeseries.append(regret_ts)

                    # Check and store if commitment was optimal
                    optimal_arm = experiment.bandit.get_optimal_arm()
                    committed_arm = experiment.commitment_arm_selected
                    run_optimal_commits.append(committed_arm == optimal_arm)

                except Exception as e:
                    print(f"    Run {run_idx+1}/{num_runs} FAILED: {e}")
                    run_regrets.append(float('nan'))
                    # Append dummy data or handle differently for time series/commit?
                    # For now, append NaN equivalents to keep lists aligned, filter later
                    run_regret_timeseries.append(np.full(T + N, float('nan')))
                    run_optimal_commits.append(None) # Use None to indicate failed run

                # Progress update within runs
                if (run_idx + 1) % 10 == 0 or run_idx == num_runs - 1:
                    print(f"    ...completed run {run_idx + 1}/{num_runs}")

            # --- Aggregate results for this strategy combination ---
            # Final Regret Stats (mean/std)
            valid_final_regrets = np.array(run_regrets)[~np.isnan(run_regrets)]
            if len(valid_final_regrets) > 0:
                final_regret_results[full_name] = {
                    "mean": np.mean(valid_final_regrets),
                    "std": np.std(valid_final_regrets)
                }
                print(f"    Avg Final Regret: {final_regret_results[full_name]['mean']:.2f} ± {final_regret_results[full_name]['std']:.2f}")
            else:
                final_regret_results[full_name] = {"mean": float('inf'), "std": 0}
                print(f"    Avg Final Regret: N/A (all runs failed)")


            # Regret Time Series (convert list of arrays to 2D array, handle NaNs if needed)
            # Ensure all time series have the same length (T+N) before stacking
            valid_ts = [ts for ts in run_regret_timeseries if ts.shape == (T+N,)]
            if valid_ts:
                 regret_timeseries_results[full_name] = np.stack(valid_ts, axis=0) # Shape (num_valid_runs, T+N)
            else:
                 regret_timeseries_results[full_name] = np.array([]) # Indicate no valid data


            # Optimal Commitment Probability
            valid_commits = [c for c in run_optimal_commits if c is not None]
            if valid_commits:
                optimal_commit_results[full_name] = valid_commits # Store the list of True/False
                prob = np.mean(valid_commits) if valid_commits else 0
                print(f"    Optimal Commit Probability: {prob:.3f}")
            else:
                optimal_commit_results[full_name] = []
                print(f"    Optimal Commit Probability: N/A (all runs failed or T=0/N=0)")


    end_time_comparison = time.time()
    print(f"\nComparison Phase took {end_time_comparison - start_time_comparison:.2f} seconds.")

    # --- 3. Generate Comparison Visualizations ---
    print("\n[Phase 3: Generating Comparison Visualizations]")
    # a) Final Average Regret Comparison (Bar Chart)
    visualize_strategy_comparison(
        final_regret_results, T, N, k, num_runs,
        filename_prefix="comparison_final_regret"
    )

    # b) Average Regret Over Time (Line Plot)
    visualize_average_regret_over_time(
        regret_timeseries_results, T, N, k, num_runs,
        filename_prefix="comparison_avg_regret_time"
    )

    # c) Optimal Commitment Probability (Bar Chart)
    visualize_optimal_commitment_probability(
        optimal_commit_results, T, N, k, num_runs,
        filename_prefix="comparison_opt_commit_prob"
    )
    print("-" * 30)


    # --- 4. Optuna Hyperparameter Optimization Example ---
    print("\n[Phase 4: Optuna Hyperparameter Optimization Example]")
    # Choose a strategy and commitment policy for optimization
    strategy_to_optimize = CombinedStrategies.ucb
    strategy_name_opt = "UCB"
    commitment_strategy_opt = CommitmentStrategies.best_empirical

    # Create objective function using the factory
    objective_func = objective_factory(
        strategy_class=strategy_to_optimize,
        commitment_strategy=commitment_strategy_opt,
        num_runs=max(10, num_runs // 5), # Fewer runs for faster optimization
        T=T, N=N, k=k
    )

    # Run Optuna
    best_params = run_optuna_optimization(
         strategy_name=strategy_name_opt,
         objective_func=objective_func,
         n_trials=n_trials_optuna
    )

    # Visualize result using optimized parameters (if found)
    if best_params:
        print(f"\nRunning single experiment with optimized {strategy_name_opt} parameters: {best_params}")
        # Create the optimized policy function
        optimized_policy = lambda exp: strategy_to_optimize(exp, **best_params)
        # Dynamic name generation based on optimized params (e.g., UCB's 'c')
        opt_param_str = ", ".join(f"{k}={v:.2f}" for k, v in best_params.items())
        exp_name_opt = f"{strategy_name_opt}({opt_param_str})"
        com_name_opt = "Commit:BestEmpirical" # Assuming fixed commit strategy here

        experiment_optimized = BanditExperiment(k=k, T=T, N=N) # Use fresh bandit
        experiment_optimized.run_experiment_phase(optimized_policy)
        experiment_optimized.choose_commitment_arm(commitment_strategy_opt)
        experiment_optimized.run_commitment_phase()
        # Use updated visualization function
        visualize_single_run_details(
             experiment_optimized, exp_name_opt, com_name_opt,
             filename_prefix="optimized_run"
         )
    else:
         print("Skipping visualization for optimized parameters as optimization failed or yielded no result.")

    end_time_main = time.time()
    print("\n" + "="*40)
    print(f"Script finished successfully in {end_time_main - start_time_main:.2f} seconds.")
    print("="*40)


if __name__ == "__main__":
    # Configure experiment parameters here
    # Example: Short run for testing
    # main(T=100, N=50, k=5, num_runs=10, n_trials_optuna=10)
    # Example: Longer run like original
    main(T=1000, N=500, k=10, num_runs=50, n_trials_optuna=50)