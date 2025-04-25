# File: visualization.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 

# Set seaborn style
sns.set_style('whitegrid')

# Configure matplotlib fonts (adjust if needed for your system)
try:
    plt.rcParams['font.family'] = ['Arial Unicode MS', 'Heiti TC', 'Songti SC', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
except Exception as e:
    print(f"Font setting failed: {e}. Using default.")

plt.rcParams['font.size'] = 12
plt.rcParams['axes.formatter.use_mathtext'] = True
plt.rcParams['axes.titlesize'] = 16 # Slightly larger titles
plt.rcParams['axes.labelsize'] = 14 # Slightly larger labels
plt.rcParams['xtick.labelsize'] = 12 # Slightly larger ticks
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 11

# --- (Keep Original visualize_experiment - Refined) ---
def visualize_single_run_details(experiment, strategy_name="Unknown Strategy", commitment_name="Unknown Commitment", filename_prefix="single_run"):
    """
    Visualize detailed results from a single experiment run (Refined).

    Args:
        experiment: Completed BanditExperiment object.
        strategy_name: Name of the experiment phase strategy.
        commitment_name: Name of the commitment phase strategy.
        filename_prefix: Prefix for the output image file.
    """
    experiment_arms, experiment_rewards = experiment.get_experiment_data()
    commitment_arm, commitment_rewards = experiment.get_commitment_data()
    k = experiment.k
    T = experiment.T
    N = experiment.N
    true_means = experiment.bandit.get_true_means()
    arm_counts = experiment.get_arm_counts()
    arm_values = experiment.get_arm_values() # Estimated values at end of T
    optimal_arm_idx = experiment.bandit.get_optimal_arm()

    fig, axes = plt.subplots(2, 2, figsize=(16, 13)) # Adjusted figsize
    ax1, ax2, ax3, ax4 = axes.flatten()
    fig.suptitle(f'Single Run: Exp="{strategy_name}", Commit="{commitment_name}"\n(T={T}, N={N}, k={k})', fontsize=18)

    # 1. Experiment Phase: Arm Selection Counts
    sns.barplot(x=list(range(k)), y=arm_counts, ax=ax1, palette='viridis', order=list(range(k)))
    ax1.set_title(f'Experiment Phase (T={T}): Arm Pull Counts')
    ax1.set_xlabel('Arm Index')
    ax1.set_ylabel('Number of Pulls')
    ax1.tick_params(axis='x', rotation=0)
    if commitment_arm is not None:
      ax1.get_xticklabels()[commitment_arm].set_weight('bold') # Bold committed arm
      ax1.get_xticklabels()[commitment_arm].set_color('blue')
    ax1.get_xticklabels()[optimal_arm_idx].set_color('red') # Red optimal arm label


    # 2. Estimated vs True Mean Rewards (at end of T)
    x = np.arange(k)
    width = 0.35
    rects1 = ax2.bar(x - width/2, arm_values, width, label=f'Estimated Mean (at T={T})', color='skyblue')
    rects2 = ax2.bar(x + width/2, true_means, width, label='True Mean', color='salmon', alpha=0.8)
    ax2.axvline(optimal_arm_idx + width/2, color='red', linestyle='--', linewidth=2, label=f'Optimal Arm ({optimal_arm_idx})')
    if commitment_arm is not None:
        ax2.axvline(commitment_arm - width/2, color='blue', linestyle=':', linewidth=2, label=f'Committed Arm ({commitment_arm})')
    ax2.set_title('Mean Rewards: Estimated vs. True')
    ax2.set_xlabel('Arm Index')
    ax2.set_ylabel('Mean Reward')
    ax2.set_xticks(x)
    ax2.set_xticklabels(x)
    ax2.legend(fontsize=10)
    ax2.tick_params(axis='x', rotation=0)
    if commitment_arm is not None:
       ax2.get_xticklabels()[commitment_arm].set_weight('bold')
       ax2.get_xticklabels()[commitment_arm].set_color('blue')
    ax2.get_xticklabels()[optimal_arm_idx].set_color('red')


    # 3. Experiment Phase: Cumulative Reward
    if T > 0 and experiment_rewards:
        cumulative_rewards_exp = np.cumsum(experiment_rewards)
        sns.lineplot(x=range(1, T + 1), y=cumulative_rewards_exp, ax=ax3, color='green')
        ax3.set_title(f'Experiment Phase (T={T}): Cumulative Reward')
        ax3.set_xlabel('Round (1 to T)')
        ax3.set_ylabel('Cumulative Reward')
        ax3.grid(True, linestyle='--', alpha=0.6)
    else:
        ax3.text(0.5, 0.5, 'No Experiment Phase (T=0)', horizontalalignment='center', verticalalignment='center', transform=ax3.transAxes)
        ax3.set_title('Experiment Phase (T=0)')


    # 4. Commitment Phase: Cumulative Reward
    if N > 0 and commitment_rewards and commitment_arm is not None:
        cumulative_rewards_com = np.cumsum(commitment_rewards)
        sns.lineplot(x=range(T + 1, T + N + 1), y=cumulative_rewards_com, ax=ax4, color='purple')
        ax4.set_title(f'Commitment Phase (N={N}, Arm={commitment_arm}): Cumulative Reward')
        ax4.set_xlabel(f'Round ({T+1} to {T+N})')
        ax4.set_ylabel('Cumulative Reward during Commitment')
        # Add line for optimal reward during commitment phase
        optimal_commit_reward = experiment.bandit.get_true_means()[commitment_arm] * np.arange(1, N + 1)
        ax4.plot(range(T + 1, T + N + 1), optimal_commit_reward, linestyle=':', color='gray', label=f'Expected Reward if Arm {commitment_arm} Pulled')
        ax4.legend(fontsize=9)
        ax4.grid(True, linestyle='--', alpha=0.6)
    else:
        ax4.text(0.5, 0.5, 'No Commitment Phase (N=0)', horizontalalignment='center', verticalalignment='center', transform=ax4.transAxes)
        ax4.set_title('Commitment Phase (N=0)')

    plt.tight_layout(rect=[0, 0.03, 1, 0.93]) # Adjust layout
    clean_strategy_name = "".join(c if c.isalnum() else "_" for c in strategy_name)
    clean_commit_name = "".join(c if c.isalnum() else "_" for c in commitment_name)
    filename = f'{filename_prefix}_{clean_strategy_name}_{clean_commit_name}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  Single run details saved to: {filename}")
    plt.close(fig)


# --- (Keep Original visualize_comparison - Refined) ---
def visualize_strategy_comparison(results_dict, T, N, k, num_runs, filename_prefix="comparison"):
    """
    Visualize comparison of different strategies based on final average total regret (Refined).

    Args:
        results_dict: Dict where keys are strategy names and values are dicts {'mean': avg_regret, 'std': std_regret}.
        T: Experiment phase rounds.
        N: Commitment phase rounds.
        k: Number of arms.
        num_runs: Number of runs averaged over.
        filename_prefix: Prefix for the output image file.
    """
    if not results_dict:
        print("Warning: No results found in comparison dictionary. Skipping plot.")
        return

    # Convert to DataFrame for easier sorting/plotting
    df = pd.DataFrame.from_dict(results_dict, orient='index')
    df = df.sort_values(by='mean')

    # Create figure
    n_strategies = len(df)
    fig, ax = plt.subplots(figsize=(max(12, n_strategies * 0.7), 8)) # Dynamic width

    # Bar plot
    colors = sns.color_palette("viridis", n_strategies)
    bars = ax.bar(df.index, df['mean'], yerr=df['std'], capsize=5, color=colors)

    ax.set_ylabel('Average Total Regret (Lower is Better)')
    ax.set_xlabel('Strategy Combination (Experiment Policy + Commitment Policy)')
    ax.set_title(f'Strategy Comparison: Average Total Regret\n(T={T}, N={N}, k={k}, Runs={num_runs})')
    ax.tick_params(axis='x', rotation=45, )

    # Add value labels on bars
    ax.bar_label(bars, fmt='%.2f', padding=3, fontsize=10)

    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    filename = f'{filename_prefix}_T{T}_N{N}_k{k}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  Strategy comparison saved to: {filename}")
    plt.close(fig)


# --- NEW VISUALIZATION FUNCTIONS ---

def visualize_average_regret_over_time(regret_data, T, N, k, num_runs, filename_prefix="avg_regret_time"):
    """
    Visualize average cumulative regret over time for different strategies.

    Args:
        regret_data: Dict where keys are strategy names and values are numpy arrays
                     of shape (num_runs, T + N), containing cumulative regret at each step.
        T: Experiment phase rounds.
        N: Commitment phase rounds.
        k: Number of arms.
        num_runs: Number of runs averaged over.
        filename_prefix: Prefix for the output image file.
    """
    if not regret_data:
        print("Warning: No regret time-series data provided. Skipping plot.")
        return

    plt.figure(figsize=(14, 8))
    total_rounds = T + N
    rounds = np.arange(1, total_rounds + 1)

    # Define a color palette
    colors = sns.color_palette("husl", len(regret_data))
    color_map = {name: color for name, color in zip(regret_data.keys(), colors)}

    # Sort strategies by final mean regret for legend order
    final_mean_regrets = {name: np.mean(data[:, -1]) for name, data in regret_data.items()}
    sorted_names = sorted(regret_data.keys(), key=lambda name: final_mean_regrets[name])

    for name in sorted_names:
        data = regret_data[name] # Shape: (num_runs, T+N)
        mean_regret = np.mean(data, axis=0)
        std_regret = np.std(data, axis=0)

        # Plot mean regret line
        plt.plot(rounds, mean_regret, label=f"{name} (Final Avg: {mean_regret[-1]:.2f})", color=color_map[name], linewidth=2)

        # Plot confidence interval (optional, can make plot busy)
        # plt.fill_between(rounds, mean_regret - std_regret, mean_regret + std_regret,
        #                  color=color_map[name], alpha=0.1)

    # Add vertical line separating Experiment and Commitment phases
    if T > 0 and N > 0:
        plt.axvline(x=T, color='gray', linestyle=':', linewidth=2, label=f'End of Experiment Phase (T={T})')

    plt.title(f'Average Cumulative Regret Over Time\n(T={T}, N={N}, k={k}, Runs={num_runs})')
    plt.xlabel('Round Number (1 to T+N)')
    plt.ylabel('Average Cumulative Regret')
    plt.legend(loc='upper left', fontsize=10) # Adjust location as needed
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim(1, total_rounds)
    plt.ylim(bottom=0) # Regret should not be negative

    plt.tight_layout()
    filename = f'{filename_prefix}_T{T}_N{N}_k{k}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  Average regret over time saved to: {filename}")
    plt.close()


def visualize_optimal_commitment_probability(commitment_data, T, N, k, num_runs, filename_prefix="opt_commit_prob"):
    """
    Visualize the probability that the committed arm was the optimal arm.

    Args:
        commitment_data: Dict where keys are strategy names and values are lists
                         (length num_runs) of booleans indicating if committed arm was optimal.
                         e.g., {'Strategy A': [True, False, True,...], 'Strategy B': [False, False, True,...]}
        T: Experiment phase rounds.
        N: Commitment phase rounds.
        k: Number of arms.
        num_runs: Number of runs averaged over.
        filename_prefix: Prefix for the output image file.
    """
    if not commitment_data:
        print("Warning: No commitment optimality data provided. Skipping plot.")
        return

    probabilities = {}
    for name, data in commitment_data.items():
        if len(data) > 0:
             probabilities[name] = np.mean(data) # Calculate probability (mean of True/False list)
        else:
             probabilities[name] = 0.0

    # Convert to DataFrame for plotting
    df = pd.DataFrame.from_dict(probabilities, orient='index', columns=['Probability'])
    df = df.sort_values(by='Probability', ascending=False)

    # Create figure
    n_strategies = len(df)
    fig, ax = plt.subplots(figsize=(max(10, n_strategies * 0.6), 7)) # Dynamic width

    # Bar plot
    colors = sns.color_palette("coolwarm", n_strategies)
    bars = ax.bar(df.index, df['Probability'], color=colors)

    ax.set_ylabel('Probability of Committing to True Optimal Arm')
    ax.set_xlabel('Strategy Combination (Experiment Policy + Commitment Policy)')
    ax.set_title(f'Commitment Phase Performance: Optimal Arm Selection Rate\n(T={T}, N={N}, k={k}, Runs={num_runs})')
    ax.set_ylim(0, 1.05) # Probability range 0 to 1
    ax.tick_params(axis='x', rotation=45, )

    # Add value labels on bars
    ax.bar_label(bars, fmt='%.3f', padding=3, fontsize=10)

    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    filename = f'{filename_prefix}_T{T}_N{N}_k{k}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  Optimal commitment probability saved to: {filename}")
    plt.close(fig)