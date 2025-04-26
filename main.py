import os
import numpy as np
import matplotlib.pyplot as plt
import time

from experiment import BanditExperiment
from strategies import CombinedStrategies, CommitmentStrategies, DynamicExplorationStrategies
from visualization import (
    visualize_single_run_details,
    visualize_strategy_comparison,
    visualize_average_regret_over_time,
    visualize_tn_relation,
    visualize_dynamic_strategies,
    visualize_tn_impact_on_exploration
)
from optimization import objective_factory, run_optuna_optimization

# Helper function to calculate regret per step
def calculate_cumulative_regret_per_step(experiment):
    """计算每一步的累积后悔值"""
    T = experiment.T
    N = experiment.N
    total_rounds = T + N
    if total_rounds == 0:
        return np.array([])

    optimal_mean = experiment.bandit.get_optimal_mean()
    true_means = experiment.bandit.get_true_means()

    # 合并两个阶段的臂选择和奖励
    arms_chosen = experiment.experiment_arms_chosen.copy()
    rewards_received = experiment.experiment_rewards_received.copy()
    if N > 0 and experiment.commitment_arm_selected is not None:
        commitment_arm = experiment.commitment_arm_selected
        arms_chosen.extend([commitment_arm] * N)
        rewards_received.extend(experiment.commitment_rewards_received)

    cumulative_regret = np.zeros(total_rounds)
    
    for t in range(total_rounds):
        if t < len(arms_chosen):
            chosen_arm = arms_chosen[t]
            expected_reward_chosen_arm = true_means[chosen_arm]
            instantaneous_regret = optimal_mean - expected_reward_chosen_arm

            if t == 0:
                cumulative_regret[t] = instantaneous_regret
            else:
                cumulative_regret[t] = cumulative_regret[t-1] + instantaneous_regret

    return cumulative_regret

def run_tn_analysis(experiment_policies, commitment_policies, k=10, budget=1500, t_values=None, n_values=None, num_runs=20):
    """
    分析不同T和N组合对后悔值的影响
    
    参数:
        experiment_policies: 实验策略字典
        commitment_policies: 承诺策略字典
        k: 臂数
        budget: 总轮数上限(T+N)
        t_values: 要测试的T值列表（如果None，自动生成）
        n_values: 要测试的N值列表（如果None，自动生成）
        num_runs: 每个配置的运行次数
    """
    print(f"\n[T/N Analysis] budget={budget}, k={k}, runs={num_runs}")
    
    # 如果没有指定T和N值，生成一些合理的测试点
    if t_values is None or n_values is None:
        # 将总预算分成不同的T/N比例
        ratios = [0.1, 0.2, 0.33, 0.5, 0.67, 0.8, 0.9]
        t_values = [int(budget * r) for r in ratios]
        n_values = [budget - t for t in t_values]
    
    # 存储每种策略组合的结果
    all_results = {}  # 策略名 -> [(T, N, 后悔值), ...]
    
    # 计算要执行的总组合数
    total_combinations = len(experiment_policies) * len(commitment_policies) * len(t_values)
    combination_counter = 0
    
    # 对每种策略组合运行实验
    for exp_name, exp_policy in experiment_policies.items():
        for com_name, com_policy in commitment_policies.items():
            strategy_name = f"{exp_name}+{com_name}"
            results = []
            
            for i, (t, n) in enumerate(zip(t_values, n_values)):
                combination_counter += 1
                print(f"Running combination {combination_counter}/{total_combinations}: {strategy_name}, T={t}, N={n}")
                
                # 记录该配置的后悔值
                regrets = []
                for run in range(num_runs):
                    try:
                        # 创建并运行实验
                        experiment = BanditExperiment(k=k, T=t, N=n)
                        experiment.run_experiment_phase(exp_policy)
                        experiment.choose_commitment_arm(com_policy)
                        experiment.run_commitment_phase()
                        regret = experiment.calculate_regret()
                        regrets.append(regret)
                    except Exception as e:
                        print(f"  Run {run+1} failed: {e}")
                
                # 计算平均后悔值
                if regrets:
                    avg_regret = np.mean(regrets)
                    std_regret = np.std(regrets)
                    print(f"  Average regret: {avg_regret:.2f} ± {std_regret:.2f}")
                    results.append((t, n, avg_regret))
            
            # 存储该策略的所有结果
            if results:
                all_results[strategy_name] = results
    
    # 可视化每个策略的T/N关系
    for strategy_name, results in all_results.items():
        visualize_tn_relation({strategy_name: results}, k, num_runs)
    
    return all_results

def run_dynamic_strategy_test(commitment_policy, k=10, n_values=[100, 500, 1000], max_t=2000, num_runs=20):
    """
    测试动态探索策略在不同N值下的性能，并与固定T策略比较
    
    参数:
        commitment_policy: 使用的承诺策略函数
        k: 臂数
        n_values: 要测试的N值列表
        max_t: 最大允许探索轮数
        num_runs: 每个配置的运行次数
    """
    print(f"\n[动态策略测试] k={k}, 最大T={max_t}, 运行次数={num_runs}")
    print("="*40)
    
    # 创建结果存储目录
    os.makedirs("dynamic_analysis", exist_ok=True)
    
    # 定义动态策略实例
    dynamic_strategies = {
        "Hoeffding(δ=1/N)": lambda: DynamicExplorationStrategies.HoeffdingBasedExploration(
            delta_factor=1.0, max_explore=max_t
        ),
        "Hoeffding(δ=0.5/N)": lambda: DynamicExplorationStrategies.HoeffdingBasedExploration(
            delta_factor=0.5, max_explore=max_t
        ),
        "Bayesian(τ=1-1/N)": lambda: DynamicExplorationStrategies.BayesianExploration(
            threshold_factor=1.0, max_explore=max_t
        )
    }
    
    # 定义固定T策略
    fixed_strategies = {
        "UCB(c=2.0)": lambda exp: CombinedStrategies.ucb(exp, c=2.0),
        "ThompsonSampling": CombinedStrategies.thompson_sampling
    }
    
    # 存储结果
    all_results = {}  # (策略类型, 策略名, N) -> (平均后悔值, 平均探索轮数, 后悔值标准差)
    exploration_results = {}  # (策略名, N) -> (平均后悔值, 平均探索轮数, 后悔值标准差)
    
    # 对每个N值运行测试
    for n in n_values:
        print(f"\n测试N={n}:")
        
        # 1. 测试动态策略
        for strategy_name, strategy_factory in dynamic_strategies.items():
            print(f"  运行 {strategy_name}")
            regrets = []
            exploration_rounds = []
            
            for run in range(num_runs):
                try:
                    # 创建新的策略实例
                    strategy = strategy_factory()
                    
                    # 创建实验
                    experiment = BanditExperiment(k=k, T=max_t, N=n)
                    
                    # 运行动态探索
                    t = 0
                    while t < max_t and not strategy.stopped:
                        arm = strategy.choose_arm(experiment)
                        reward = experiment.bandit.pull(arm)
                        
                        # 更新实验数据
                        experiment.experiment_arms_chosen.append(arm)
                        experiment.experiment_rewards_received.append(reward)
                        experiment.arm_pull_counts[arm] += 1
                        experiment.arm_cumulative_rewards[arm] += reward
                        if experiment.arm_pull_counts[arm] > 0:
                            experiment.arm_estimated_values[arm] = (
                                experiment.arm_cumulative_rewards[arm] / experiment.arm_pull_counts[arm]
                            )
                        
                        t += 1
                    
                    # 记录实际探索轮数
                    actual_t = t
                    exploration_rounds.append(actual_t)
                    
                    # 更新实验的真实T值
                    experiment.T = actual_t
                    
                    # 选择承诺臂
                    if strategy.stopped and strategy.committed_arm is not None:
                        experiment.commitment_arm_selected = strategy.committed_arm
                    else:
                        experiment.choose_commitment_arm(commitment_policy)
                    
                    # 运行承诺阶段
                    experiment.run_commitment_phase()
                    regret = experiment.calculate_regret()
                    regrets.append(regret)
                
                except Exception as e:
                    print(f"    运行 {run+1} 出错: {e}")
            
            if regrets:
                avg_regret = np.mean(regrets)
                std_regret = np.std(regrets)
                avg_t = np.mean(exploration_rounds)
                print(f"    平均后悔值: {avg_regret:.2f} ± {std_regret:.2f}, 平均探索轮数: {avg_t:.1f}")
                
                all_results[('dynamic', strategy_name, n)] = (avg_regret, avg_t, std_regret)
                exploration_results[(strategy_name, n)] = (avg_regret, avg_t, std_regret)
        
        # 2. 测试固定T策略
        t_values = [10, 50, 100, 200, 500, 1000]
        t_values = [t for t in t_values if t <= max_t]
        
        for strategy_name, strategy_func in fixed_strategies.items():
            for t in t_values:
                print(f"  运行 {strategy_name} T={t}")
                regrets = []
                
                for run in range(num_runs):
                    try:
                        experiment = BanditExperiment(k=k, T=t, N=n)
                        experiment.run_experiment_phase(strategy_func)
                        experiment.choose_commitment_arm(commitment_policy)
                        experiment.run_commitment_phase()
                        regret = experiment.calculate_regret()
                        regrets.append(regret)
                    except Exception as e:
                        print(f"    运行 {run+1} 出错: {e}")
                
                if regrets:
                    avg_regret = np.mean(regrets)
                    std_regret = np.std(regrets)
                    print(f"    平均后悔值: {avg_regret:.2f} ± {std_regret:.2f}")
                    
                    all_results[('fixed', f"{strategy_name} (T={t})", n)] = (avg_regret, t, std_regret)
    
    # 可视化结果
    visualize_dynamic_strategies(all_results, k, num_runs, filename_prefix="dynamic_analysis/comparison")
    visualize_tn_impact_on_exploration(exploration_results, filename_prefix="dynamic_analysis/n_impact")
    
    print("\n[动态策略测试完成]")
    return all_results, exploration_results


def main(T=1000, N=500, k=10, num_runs=50, n_trials_optuna=50, run_tn_analysis_flag=True):
    """
    主执行函数
    """
    start_time = time.time()
    print("="*40)
    print(" K-armed Bandit Experiment - Two-Phase Framework")
    print(f" Parameters: T={T}, N={N}, k={k}, num_runs={num_runs}")
    print("="*40)

    # 定义策略组合
    experiment_policies = {
        "EpsGreedy(0.1)": lambda exp: CombinedStrategies.epsilon_greedy(exp, epsilon=0.1),
        "UCB(c=2.0)": lambda exp: CombinedStrategies.ucb(exp, c=2.0),
        "ThompsonSampling": CombinedStrategies.thompson_sampling,
        "Softmax(0.1)": lambda exp: CombinedStrategies.softmax(exp, temperature=0.1)
    }

    commitment_policies = {
        "BestEmpirical": CommitmentStrategies.best_empirical,
        "MostPulled": CommitmentStrategies.most_pulled,
        "LCB(95%)": lambda exp: CommitmentStrategies.confidence_based(exp, confidence_level=0.95)
    }

    # --- 1. 单次运行示例 ---
    print("\n[Phase 1: Single Run Example]")
    # 选择一种策略组合进行演示
    exp_policy_name = "UCB(c=2.0)"
    com_policy_name = "BestEmpirical"
    exp_policy = experiment_policies[exp_policy_name]
    com_policy = commitment_policies[com_policy_name]

    print(f"Running {exp_policy_name} + {com_policy_name}")
    experiment = BanditExperiment(k=k, T=T, N=N)
    experiment.run_experiment_phase(exp_policy)
    experiment.choose_commitment_arm(com_policy)
    experiment.run_commitment_phase()
    # 可视化单次运行结果
    visualize_single_run_details(
        experiment, exp_policy_name, com_policy_name, "single_run_example"
    )

    # --- 2. 策略比较 (多次运行) ---
    print("\n[Phase 2: Strategy Comparison]")
    # 存储结果的字典
    final_regret_results = {}  # 最终平均后悔值
    regret_timeseries_results = {}  # 随时间变化的后悔值
    
    # 统计要运行的总组合数
    total_combinations = len(experiment_policies) * len(commitment_policies)
    combination_counter = 0
    
    for com_name, com_policy in commitment_policies.items():
        for exp_name, exp_policy in experiment_policies.items():
            combination_counter += 1
            full_name = f"{exp_name} + {com_name}"
            print(f"Running combination {combination_counter}/{total_combinations}: {full_name}")
            
            # 存储该组合的多次运行结果
            run_regrets = []
            run_regret_timeseries = []
            
            for run in range(num_runs):
                # 运行实验
                experiment = BanditExperiment(k=k, T=T, N=N)
                experiment.run_experiment_phase(exp_policy)
                experiment.choose_commitment_arm(com_policy)
                experiment.run_commitment_phase()
                
                # 计算并存储最终后悔值
                final_regret = experiment.calculate_regret()
                run_regrets.append(final_regret)
                
                # 计算并存储每步后悔值
                regret_ts = calculate_cumulative_regret_per_step(experiment)
                run_regret_timeseries.append(regret_ts)
                
                # 显示进度
                if (run + 1) % 10 == 0:
                    print(f"  Completed run {run + 1}/{num_runs}")
            
            # 汇总结果
            valid_regrets = np.array(run_regrets)
            final_regret_results[full_name] = {
                "mean": np.mean(valid_regrets),
                "std": np.std(valid_regrets)
            }
            print(f"  Average regret: {final_regret_results[full_name]['mean']:.2f} ± {final_regret_results[full_name]['std']:.2f}")
            
            # 存储时间序列数据
            regret_timeseries_results[full_name] = np.stack(run_regret_timeseries)
    
    # --- 3. 生成比较可视化 ---
    print("\n[Phase 3: Generating Visualizations]")
    # 最终平均后悔值比较
    visualize_strategy_comparison(
        final_regret_results, T, N, k, num_runs, "comparison_final_regret"
    )
    
    # 平均后悔值随时间变化
    visualize_average_regret_over_time(
        regret_timeseries_results, T, N, k, num_runs, "comparison_regret_over_time"
    )
    
    # --- 4. 参数优化示例 ---
    print("\n[Phase 4: Parameter Optimization Example]")
    # 为UCB策略进行参数优化
    strategy_to_optimize = CombinedStrategies.ucb
    strategy_name = "UCB"
    commitment_strategy = CommitmentStrategies.best_empirical
    
    # 创建目标函数
    objective_func = objective_factory(
        strategy_class=strategy_to_optimize,
        commitment_strategy=commitment_strategy,
        num_runs=max(5, num_runs // 10),  # 减少运行次数加快优化
        T=T, N=N, k=k
    )
    
    # 运行Optuna
    best_params = run_optuna_optimization(
        strategy_name=strategy_name,
        objective_func=objective_func,
        n_trials=n_trials_optuna
    )
    
    # 使用优化后的参数运行一次实验并可视化
    if best_params:
        print(f"\nRunning with optimized parameters: {best_params}")
        # 创建优化后的策略函数
        optimized_policy = lambda exp: strategy_to_optimize(exp, **best_params)
        # 生成策略名称
        opt_param_str = ", ".join(f"{k}={v:.2f}" for k, v in best_params.items())
        opt_strategy_name = f"{strategy_name}({opt_param_str})"
        
        # 运行实验
        experiment = BanditExperiment(k=k, T=T, N=N)
        experiment.run_experiment_phase(optimized_policy)
        experiment.choose_commitment_arm(commitment_strategy)
        experiment.run_commitment_phase()
        
        # 可视化结果
        visualize_single_run_details(
            experiment, opt_strategy_name, "BestEmpirical", "optimized_run"
        )
    
    # --- 5. T/N关系分析 (可选) ---
    if run_tn_analysis_flag:
        print("\n[Phase 5: T/N Relationship Analysis]")
        # 为了减少计算量，只选择一部分策略
        tn_experiment_policies = {
            "EpsGreedy(0.1)": lambda exp: CombinedStrategies.epsilon_greedy(exp, epsilon=0.1),
            "UCB(c=2.0)": lambda exp: CombinedStrategies.ucb(exp, c=2.0),
            "ThompsonSampling": CombinedStrategies.thompson_sampling
        }
        
        tn_commitment_policies = {
            "BestEmpirical": CommitmentStrategies.best_empirical
        }
        
        # 运行T/N分析
        tn_results = run_tn_analysis(
            experiment_policies=tn_experiment_policies,
            commitment_policies=tn_commitment_policies,
            k=k,
            budget=T+N,  # 总预算保持与主实验相同
            num_runs=max(5, num_runs // 10)  # 减少运行次数
        )
        
        print("\n[Phase 6: Dynamic Exploration Strategy Test]")
        # 运行动态策略测试
        dynamic_results, exploration_impact = run_dynamic_strategy_test(
            commitment_policy=CommitmentStrategies.best_empirical,
            k=k,
            n_values=[N//2, N, N*2],  # 测试不同的N值
            max_t=T*2,  # 最大允许探索轮数
            num_runs=max(5, num_runs // 10)  # 减少运行次数以加快分析
        )
    
    end_time = time.time()
    print("\n" + "="*40)
    print(f"Script finished in {end_time - start_time:.2f} seconds.")
    print("="*40)


if __name__ == "__main__":
    # 选择是运行快速测试还是完整实验
    RUN_QUICK_TEST = True  # 设置为False以运行完整实验
    
    if RUN_QUICK_TEST:
        # 快速测试版本
        main(T=50, N=25, k=5, num_runs=5, n_trials_optuna=5, run_tn_analysis_flag=True)
    else:
        # 完整实验版本
        main(T=1000, N=500, k=10, num_runs=50, n_trials_optuna=50, run_tn_analysis_flag=True)
