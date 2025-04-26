# File: bandit.py
import numpy as np

class KArmedBandit:
    r"""
    K臂赌博机环境。

    模拟一个有 K 个臂的赌博机，每个臂有其固定的真实平均奖励。
    拉动一个臂会返回一个随机奖励。
    """
    def __init__(self, k=10, mean_range=(0, 1), reward_std_dev=1.0, reward_type='truncated_normal'):
        r"""
        初始化K臂赌博机

        参数:
        k: 赌博机的臂数 (int > 0)
        mean_range: 每个臂的真实奖励均值 $\mu_i$ 的抽样范围 (tuple, e.g., (0, 1))
        reward_std_dev: 生成奖励时使用的标准差 (float >= 0)。仅用于 'truncated_normal' 类型。
        reward_type: 奖励生成方式 ('truncated_normal', 'bernoulli', 'beta')
        """
        if k <= 0:
            raise ValueError("臂数 k 必须是正整数")
        if reward_std_dev < 0:
             raise ValueError("奖励标准差不能为负")

        self.k = k
        self.mean_range = mean_range
        self.reward_std_dev = reward_std_dev
        self.reward_type = reward_type.lower()
        self._validate_reward_type()

        # 初始化内部状态变量
        self.true_means = None
        self.beta_params = None # 用于存储 Beta 分布的 alpha, beta 参数
        self.optimal_arm = None
        self.optimal_mean = None
        # 在 reset 方法中实际生成均值和参数
        self.reset()

    def _validate_reward_type(self):
        """检查奖励类型是否受支持"""
        supported_types = ['truncated_normal', 'bernoulli', 'beta']
        if self.reward_type not in supported_types:
            raise ValueError(f"不支持的奖励类型: {self.reward_type}. 支持的类型: {supported_types}")
        if self.reward_type == 'beta' and not (0 <= self.mean_range[0] <= 1 and 0 <= self.mean_range[1] <= 1):
             print("警告: Beta 分布的均值应在 [0, 1] 范围内。")


    def reset(self):
        """
        重置赌博机：根据 reward_type 重新随机生成每个臂的真实参数。
        对于 'beta' 类型，生成 alpha 和 beta 参数；否则生成均值。
        """
        if self.reward_type == 'beta':
            # 为每个臂随机生成 Beta 分布的 alpha 和 beta 参数
            # 例如，从 Uniform(1, 5) 中采样，以确保参数为正且分布不会过于集中
            alpha_params = np.random.uniform(1, 5, self.k)
            beta_params = np.random.uniform(1, 5, self.k)
            self.beta_params = np.stack([alpha_params, beta_params], axis=1) # 存储为 (k, 2) 数组
            # 计算对应的真实均值 (用于比较和后悔值计算)
            self.true_means = alpha_params / (alpha_params + beta_params)
        else:
            # 对于其他分布，像以前一样生成真实均值
            self.true_means = np.random.uniform(self.mean_range[0], self.mean_range[1], self.k)
            self.beta_params = None # Beta 参数不适用

        # 记录最优臂的索引和真实奖励均值 (基于计算出的或生成的均值)
        self.optimal_arm = np.argmax(self.true_means)
        self.optimal_mean = self.true_means[self.optimal_arm]

    def pull(self, arm):
        r"""
        拉动指定的臂，根据设定的 `reward_type` 返回随机奖励。

        参数:
        arm: 要拉动的臂的索引 (0 to k-1)

        返回:
        该臂产生的随机奖励值 (float or int)
        """
        if not (0 <= arm < self.k):
            raise ValueError(f"臂的索引 {arm} 必须在 0 到 {self.k-1} 之间")

        mean = self.true_means[arm]

        if self.reward_type == 'truncated_normal':
            # 生成正态分布奖励，然后截断到 [0, 1] 范围
            reward = np.random.normal(mean, self.reward_std_dev)
            return np.clip(reward, 0, 1) # 截断到 [0, 1]
        elif self.reward_type == 'bernoulli':
            # 生成 Bernoulli 奖励 (0 或 1)，概率为 mean
            if not (0 <= mean <= 1):
                 mean = np.clip(mean, 0, 1)
            return np.random.binomial(1, mean)
        elif self.reward_type == 'beta':
            # 使用在 reset() 中为该臂生成的 alpha 和 beta 参数
            if self.beta_params is None:
                 raise RuntimeError("内部错误: reward_type 为 'beta' 但 beta_params 未在 reset() 中设置。")
            alpha, beta = self.beta_params[arm]
            return np.random.beta(alpha, beta)
        else:
             raise RuntimeError(f"内部错误: 未处理的奖励类型 {self.reward_type}")

    def get_optimal_arm(self):
        """返回最优臂的索引"""
        return self.optimal_arm

    def get_optimal_mean(self):
        """返回最优臂的真实奖励均值"""
        return self.optimal_mean

    def get_true_means(self):
        """返回所有臂的真实奖励均值列表"""
        return self.true_means
