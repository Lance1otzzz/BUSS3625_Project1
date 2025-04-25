import numpy as np
import optuna
from experiment import BanditExperiment # 确保导入 BanditExperiment
from strategies import CombinedStrategies, CommitmentStrategies # 确保导入策略

def objective_factory(strategy_class, commitment_strategy, num_runs=20, T=1000, N=100, k=10):
    """
    工厂函数，用于为不同的策略创建 Optuna 目标函数。

    参数:
    strategy_class: 来自 CombinedStrategies 的策略类或方法 (e.g., CombinedStrategies.epsilon_greedy)
    commitment_strategy: 来自 CommitmentStrategies 的承诺策略 (e.g., CommitmentStrategies.best_empirical)
    num_runs: 每次评估参数时的实验重复次数
    T: 实验阶段的轮数
    N: 承诺阶段的轮数
    k: 赌博机的臂数

    返回:
    一个 Optuna 目标函数
    """
    def objective(trial):
        params = {}
        # 根据策略动态建议超参数
        if strategy_class == CombinedStrategies.epsilon_greedy or strategy_class == CombinedStrategies.adaptive_epsilon_greedy:
            params['epsilon'] = trial.suggest_float('epsilon', 1e-5, 0.5, log=True) # Epsilon 通常在较小范围优化
            if strategy_class == CombinedStrategies.adaptive_epsilon_greedy:
                 params['base_epsilon'] = params.pop('epsilon') # 重命名为 base_epsilon
                 params['scaling_factor'] = trial.suggest_float('scaling_factor', 0.1, 1.0)
        elif strategy_class == CombinedStrategies.ucb:
            params['c'] = trial.suggest_float('c', 0.1, 5.0) # UCB 的探索参数 c
        elif strategy_class == CombinedStrategies.softmax:
            params['temperature'] = trial.suggest_float('temperature', 1e-3, 1.0, log=True) # Softmax 的温度参数
        elif strategy_class == CombinedStrategies.hybrid_ucb_thompson:
            params['weight'] = trial.suggest_float('weight', 0.0, 1.0) # UCB 权重
            params['c'] = trial.suggest_float('c', 0.1, 5.0) # UCB 探索参数

        # 创建策略函数 lambda
        policy = lambda exp: strategy_class(exp, **params)

        regrets = []
        for _ in range(num_runs):
            experiment = BanditExperiment(k=k, T=T, N=N)
            experiment.run_experiment_phase(policy)
            experiment.choose_commitment_arm(commitment_strategy)
            experiment.run_commitment_phase()
            regret = experiment.calculate_regret()
            regrets.append(regret)

        average_regret = np.mean(regrets)
        return average_regret

    return objective


def run_optuna_optimization(strategy_name, objective_func, n_trials=50):
    """
    运行 Optuna 优化过程。

    参数:
    strategy_name: 正在优化的策略名称 (用于打印信息)
    objective_func: Optuna 目标函数 (由 objective_factory 创建)
    n_trials: 优化的试验次数
    """
    print(f"\n--- 使用 Optuna 优化 {strategy_name} 策略 ---")
    # 创建 Optuna study，目标是最小化后悔值
    study = optuna.create_study(direction='minimize')
    # 运行优化
    try:
        study.optimize(objective_func, n_trials=n_trials, n_jobs=-1) # 使用所有可用 CPU 核
    except Exception as e:
         print(f"Optuna 优化过程中出现错误: {e}")
         return None # 优化失败

    print(f"优化完成。试验次数: {len(study.trials)}")
    if study.best_trial:
        print(f"最佳 {strategy_name} 超参数: {study.best_params}")
        print(f"对应的最小平均后悔值: {study.best_value:.4f}")
        return study.best_params
    else:
        print("未能找到最佳参数。")
        return None