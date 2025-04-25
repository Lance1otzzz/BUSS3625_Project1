# File: experiment.py
import numpy as np
from bandit import KArmedBandit # 确保导入 KArmedBandit

class BanditExperiment:
    """
    多臂赌博机实验类，封装了实验流程，包含明确的实验阶段和承诺阶段。
    这种结构天然支持两阶段混合策略：
    - `run_experiment_phase` 使用一个策略函数运行 T 轮探索。
    - `choose_commitment_arm` 使用另一个策略函数根据 T 轮结果选择一个臂。
    - `run_commitment_phase` 使用选定的臂运行 N 轮承诺。
    """
    def __init__(self, k=10, T=1000, N=100, bandit_params=None):
        """
        初始化实验

        参数:
        k: 赌博机的臂数
        T: 实验阶段的轮数 (必须 >= 0)
        N: 承诺阶段的轮数 (必须 >= 0)
        bandit_params: 传递给 KArmedBandit 的参数字典 (可选)
        """
        if T < 0 or N < 0:
            raise ValueError("T 和 N 必须是非负整数")
        self.k = k
        self.T = T
        self.N = N
        # 允许自定义 bandit 参数，例如均值范围
        self.bandit = KArmedBandit(k=k, **(bandit_params or {}))

        # 初始化存储实验数据的列表
        self.reset()

    def reset(self):
        """
        重置实验状态和赌博机实例。
        """
        self.bandit.reset() # 重置赌博机臂的真实均值

        # 存储实验阶段数据
        self.experiment_arms_chosen = [] # 每轮选择的臂
        self.experiment_rewards_received = [] # 每轮获得的奖励

        # 存储承诺阶段数据
        self.commitment_arm_selected = None # 选择的承诺臂
        self.commitment_rewards_received = [] # 每轮获得的奖励

        # 存储每个臂的统计信息 (在实验阶段更新)
        self.arm_pull_counts = np.zeros(self.k, dtype=int) # 每个臂被拉动的次数
        self.arm_cumulative_rewards = np.zeros(self.k, dtype=float) # 每个臂的总奖励
        self.arm_estimated_values = np.zeros(self.k, dtype=float) # 每个臂的估计平均奖励

        # 总轮次计数器
        self.total_rounds_elapsed = 0

    def run_experiment_phase(self, experiment_policy):
        """
        运行实验阶段 (T 轮)。

        参数:
        experiment_policy: 选择臂的策略函数。
                           该函数接收当前实验对象 (self) 作为参数，
                           并返回选择的臂的索引 (0 to k-1)。
        """
        if self.T == 0:
            # print("实验阶段轮数 T = 0，跳过实验阶段。") # Reduced print noise
            return # 如果 T=0，不执行任何操作

        # print(f"开始实验阶段 (T={self.T})...") # Reduced print noise
        for t in range(self.T):
            self.total_rounds_elapsed += 1
            # 使用策略选择臂
            # 策略函数可以访问 self.arm_pull_counts, self.arm_estimated_values 等信息
            arm_to_pull = experiment_policy(self)

            if not (0 <= arm_to_pull < self.k):
                 raise ValueError(f"策略返回了无效的臂索引: {arm_to_pull}")

            # 拉动选择的臂并获得奖励
            reward = self.bandit.pull(arm_to_pull)

            # 更新实验数据记录
            self.experiment_arms_chosen.append(arm_to_pull)
            self.experiment_rewards_received.append(reward)

            # 更新被选臂的统计信息 (使用增量更新公式计算均值)
            self.arm_pull_counts[arm_to_pull] += 1
            self.arm_cumulative_rewards[arm_to_pull] += reward
            # 更新估计均值: V_n = V_{n-1} + (R_n - V_{n-1}) / n
            # 或者直接计算: V_n = Sum(R_i) / n
            # 为避免潜在的浮点数问题和效率，直接计算更简单
            self.arm_estimated_values[arm_to_pull] = self.arm_cumulative_rewards[arm_to_pull] / self.arm_pull_counts[arm_to_pull]
        # print("实验阶段完成。") # Reduced print noise


    def choose_commitment_arm(self, commitment_policy):
        """
        在实验阶段结束后，选择承诺阶段要使用的臂。

        参数:
        commitment_policy: 选择承诺臂的策略函数。
                           该函数接收当前实验对象 (self) 作为参数，
                           （可以访问实验结束时的 self.arm_estimated_values, self.arm_pull_counts）
                           并返回选择的臂的索引 (0 to k-1)。
        """
        if self.T == 0 and self.N > 0:
             print("警告: 实验阶段 T=0，承诺策略可能基于零信息。")
             # 可以在 T=0 时强制随机选择或让策略自行处理
             # self.commitment_arm_selected = np.random.randint(self.k)
             # return self.commitment_arm_selected

        # print("选择承诺臂...") # Reduced print noise
        self.commitment_arm_selected = commitment_policy(self)

        if not (isinstance(self.commitment_arm_selected, (int, np.integer)) and 0 <= self.commitment_arm_selected < self.k):
             raise ValueError(f"承诺策略返回了无效的臂索引或类型: {self.commitment_arm_selected} (类型: {type(self.commitment_arm_selected)})")


        # print(f"承诺臂已选定: {self.commitment_arm_selected}") # Reduced print noise
        return self.commitment_arm_selected

    def run_commitment_phase(self):
        """
        运行承诺阶段 (N 轮)。
        必须先调用 choose_commitment_arm 选择承诺臂。
        """
        if self.N == 0:
             # print("承诺阶段轮数 N = 0，跳过承诺阶段。") # Reduced print noise
             return # 如果 N=0，不执行任何操作

        if self.commitment_arm_selected is None:
            # Check if T > 0, if T=0 commitment arm might be intentionally None if N=0 too.
            if self.T > 0 or self.N > 0: # Only raise error if phases were expected
                 raise ValueError("必须先调用 choose_commitment_arm 选择承诺臂才能运行承诺阶段。")
            else:
                 return # Both T=0 and N=0, okay to have no commitment arm

        # print(f"开始承诺阶段 (N={self.N}, 臂={self.commitment_arm_selected})...") # Reduced print noise
        for _ in range(self.N):
            self.total_rounds_elapsed += 1
            # 拉动承诺的臂并获得奖励
            reward = self.bandit.pull(self.commitment_arm_selected)
            self.commitment_rewards_received.append(reward)
        # print("承诺阶段完成。") # Reduced print noise

    def calculate_regret(self):
        """
        计算总后悔值 (Total Regret)。
        Regret = (期望最优总奖励) - (实际总奖励)

        返回:
        总后悔值 (float)
        """
        optimal_mean = self.bandit.get_optimal_mean()
        total_rounds = self.T + self.N

        if total_rounds == 0:
            return 0.0 # 没有轮次，没有后悔值

        # 期望最优总奖励 = 最优臂的平均奖励 * 总轮数
        optimal_total_expected_reward = optimal_mean * total_rounds

        # 实际总奖励 = 实验阶段总奖励 + 承诺阶段总奖励
        actual_total_reward = sum(self.experiment_rewards_received) + sum(self.commitment_rewards_received)

        # 计算后悔值
        regret = optimal_total_expected_reward - actual_total_reward
        return regret

    # --- Helper methods to access state ---
    def get_arm_values(self):
        r"""获取每个臂当前的估计平均奖励 $\hat{\mu}_i$"""
        return self.arm_estimated_values

    def get_arm_counts(self):
        """获取每个臂到目前为止被拉动的次数 $n_i$"""
        return self.arm_pull_counts

    def get_total_rewards(self):
        """获取所有轮次累计的总奖励"""
        return sum(self.experiment_rewards_received) + sum(self.commitment_rewards_received)

    def get_experiment_data(self):
        """获取实验阶段的数据 (选择的臂列表, 获得的奖励列表)"""
        return self.experiment_arms_chosen, self.experiment_rewards_received

    def get_commitment_data(self):
        """获取承诺阶段的数据 (选择的承诺臂索引, 获得的奖励列表)"""
        return self.commitment_arm_selected, self.commitment_rewards_received