# File: optimization.py
import numpy as np
import optuna
from experiment import BanditExperiment

def objective_factory(strategy_class, commitment_strategy, num_runs=20, T=1000, N=100, k=10):
    """
    工厂函数，用于为不同的策略创建 Optuna 目标函数。
    """
    def objective(trial):
        params = {}
        # 根据策略动态建议超参数
        strategy_name = strategy_class.__name__
        if 'epsilon_greedy' in strategy_name.lower():
            params['epsilon'] = trial.suggest_float('epsilon', 1e-5, 0.5, log=True)
            if 'adaptive' in strategy_name.lower():
                params['base_epsilon'] = params.pop('epsilon')
                params['scaling_factor'] = trial.suggest_float('scaling_factor', 0.1, 1.0)
        elif 'ucb' in strategy_name.lower():
            params['c'] = trial.suggest_float('c', 0.1, 5.0)
        elif 'softmax' in strategy_name.lower():
            params['temperature'] = trial.suggest_float('temperature', 1e-3, 1.0, log=True)
        elif 'hybrid' in strategy_name.lower():
            params['weight'] = trial.suggest_float('weight', 0.0, 1.0)
            if 'ucb' in strategy_name.lower():
                params['c'] = trial.suggest_float('c', 0.1, 5.0)

        # 创建策略函数
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
    """
    print(f"\n--- Optimizing {strategy_name} strategy with Optuna ---")
    # 创建 Optuna study
    study = optuna.create_study(direction='minimize')
    # 运行优化
    try:
        study.optimize(objective_func, n_trials=n_trials)
    except Exception as e:
         print(f"Optuna optimization error: {e}")
         return None

    print(f"Optimization completed. Number of trials: {len(study.trials)}")
    if study.best_trial:
        print(f"Best {strategy_name} parameters: {study.best_params}")
        print(f"Corresponding minimum average regret: {study.best_value:.4f}")
        return study.best_params
    else:
        print("Could not find best parameters.")
        return None
