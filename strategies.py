# File: strategies.py
import numpy as np
import math

class CombinedStrategies:
    """包含用于多臂赌博机问题"实验阶段"的各种策略。"""
    
    @staticmethod
    def epsilon_greedy(experiment, epsilon=0.1):
        """
        ε-贪婪策略: 以 ε 概率随机探索，以 1-ε 概率选择当前最优臂。
        """
        if np.random.random() < epsilon:
            # 探索: 随机选择一个臂
            return np.random.randint(experiment.k)
        else:
            # 利用: 选择当前估计值最高的臂
            arm_values = experiment.get_arm_values()
            # 处理所有臂估计值相同的情况
            if len(set(arm_values)) == 1:
                 return np.random.randint(experiment.k)
            # 随机选择一个最大值索引（如果有多个）
            max_indices = np.where(arm_values == np.max(arm_values))[0]
            return np.random.choice(max_indices)

    @staticmethod
    def ucb(experiment, c=2.0):
        """
        上置信界 (UCB1) 策略: 选择具有最高 UCB 指数的臂。
        """
        t = sum(experiment.get_arm_counts()) + 1  # 当前总轮次
        arm_values = experiment.get_arm_values()
        arm_counts = experiment.get_arm_counts()

        # 优先选择从未被选择过的臂
        zero_indices = np.where(arm_counts == 0)[0]
        if len(zero_indices) > 0:
            return zero_indices[0]

        # 计算 UCB 值
        if t == 1:
            exploration_bonus = np.ones(experiment.k) * np.inf
        else:
            exploration_bonus = c * np.sqrt(np.log(t) / arm_counts)
        
        ucb_values = arm_values + exploration_bonus

        # 处理多个臂具有相同最高 UCB 值的情况
        max_ucb_indices = np.where(ucb_values == np.max(ucb_values))[0]
        return np.random.choice(max_ucb_indices)

    @staticmethod
    def thompson_sampling(experiment):
        """
        汤普森采样 (Thompson Sampling) 策略 (使用 Beta 分布)。
        """
        samples = np.zeros(experiment.k)
        arm_cumulative_rewards = experiment.arm_cumulative_rewards
        arm_counts = experiment.get_arm_counts()

        for arm in range(experiment.k):
            if arm_counts[arm] == 0:
                # 对于未探索的臂，从先验 Beta(1, 1) 采样
                samples[arm] = np.random.beta(1, 1)
                continue

            # 假设reward是0-1之间的值
            successes = arm_cumulative_rewards[arm]
            failures = arm_counts[arm] - successes
            
            # Beta 分布参数 (alpha=成功+1, beta=失败+1)
            alpha = max(1, successes + 1)
            beta = max(1, failures + 1)

            # 从 Beta 后验分布中抽取样本
            samples[arm] = np.random.beta(alpha, beta)

        # 选择样本值最大的臂
        max_sample_indices = np.where(samples == np.max(samples))[0]
        return np.random.choice(max_sample_indices)

    @staticmethod
    def softmax(experiment, temperature=0.1):
        """
        Softmax (Boltzmann Exploration) 策略
        """
        if temperature <= 0:
            raise ValueError("Temperature must be positive.")

        arm_values = experiment.get_arm_values()
        # 防止溢出: 对所有值减去最大值
        stable_values = arm_values - np.max(arm_values)
        # 计算指数权重
        with np.errstate(over='ignore'):
            exp_values = np.exp(stable_values / temperature)

        # 处理 inf
        if np.isinf(exp_values).any():
            inf_indices = np.where(np.isinf(exp_values))[0]
            return np.random.choice(inf_indices)

        sum_exp_values = np.sum(exp_values)

        # 计算概率分布
        if sum_exp_values == 0 or np.isnan(sum_exp_values):
            return np.random.randint(experiment.k)
        else:
            probabilities = exp_values / sum_exp_values

        # 根据概率分布随机选择臂
        return np.random.choice(experiment.k, p=probabilities)

    @staticmethod
    def adaptive_epsilon_greedy(experiment, base_epsilon=0.1, scaling_factor=0.1):
        """
        自适应 ε-贪婪策略: ε 值随时间衰减。
        """
        t = sum(experiment.get_arm_counts())
        current_epsilon = base_epsilon / (1 + t * scaling_factor)

        # 使用计算出的 current_epsilon 调用标准 epsilon_greedy
        return CombinedStrategies.epsilon_greedy(experiment, epsilon=current_epsilon)


class CommitmentStrategies:
    """包含用于多臂赌博机问题"承诺阶段"的策略。"""
    
    @staticmethod
    def best_empirical(experiment):
        """
        选择实验阶段估计平均奖励最高的臂。
        """
        arm_values = experiment.get_arm_values()
        arm_counts = experiment.get_arm_counts()

        # 如果没有臂被拉动 (T=0)，随机选择
        if np.all(arm_counts == 0):
            return np.random.randint(experiment.k)

        # 处理估计值相同的情况
        max_indices = np.where(arm_values == np.max(arm_values))[0]
        return np.random.choice(max_indices)

    @staticmethod
    def most_pulled(experiment):
        """
        选择实验阶段被拉动次数最多的臂。
        """
        arm_counts = experiment.get_arm_counts()

        # 如果没有臂被拉动 (T=0)，随机选择
        if np.all(arm_counts == 0):
            return np.random.randint(experiment.k)

        # 处理次数相同的情况
        max_indices = np.where(arm_counts == np.max(arm_counts))[0]
        return np.random.choice(max_indices)

    @staticmethod
    def confidence_based(experiment, confidence_level=0.95):
        """
        基于置信下界 (LCB) 选择臂。
        选择具有最高 LCB 的臂，这是一种更保守的选择方法。
        """
        arm_values = experiment.get_arm_values()
        arm_counts = experiment.get_arm_counts()
        k = experiment.k
        lcb = np.full(k, -np.inf)

        # 如果没有臂被拉动 (T=0)，随机选择
        if np.all(arm_counts == 0):
            return np.random.randint(k)

        alpha = 1.0 - confidence_level

        for i in range(k):
            if arm_counts[i] > 0:
                bound_width = np.sqrt(np.log(1 / alpha) / (2 * arm_counts[i]))
                lcb[i] = arm_values[i] - bound_width

        # 处理 LCB 相同的情况
        max_lcb_indices = np.where(lcb == np.max(lcb))[0]
        return np.random.choice(max_lcb_indices)
class DynamicExplorationStrategies:
    """包含动态调整探索强度的策略，根据后续承诺阶段长度N自适应调整探索。"""
    
    class HoeffdingBasedExploration:
        """
        基于Hoeffding不等式的动态探索策略。
        当一个臂的置信下界高于所有其他臂的置信上界时停止探索。
        """
        def __init__(self, delta_factor=1.0, max_explore=None):
            self.delta_factor = delta_factor  # 置信水平系数，用于delta = 1/(N*delta_factor)
            self.max_explore = max_explore    # 最大探索轮数
            self.stopped = False
            self.committed_arm = None
            self.total_pulls = 0
            
        def reset(self):
            """重置策略状态"""
            self.stopped = False
            self.committed_arm = None
            self.total_pulls = 0
        
        def choose_arm(self, experiment):
            """选择下一个要拉动的臂，如果已经找到最优臂则停止探索"""
            if self.stopped:
                return self.committed_arm
                
            k = experiment.k
            arm_counts = experiment.get_arm_counts()
            arm_values = experiment.get_arm_values()
            N = experiment.N  # 承诺阶段长度，用于调整置信水平
            
            # 计算delta: 与N成反比关系
            delta = 1.0 / (N * self.delta_factor)
            delta = min(delta, 0.5)  # 确保delta合理
            
            # 优先选择未被拉动的臂
            zero_indices = np.where(arm_counts == 0)[0]
            if len(zero_indices) > 0:
                return zero_indices[0]
                
            # 检查是否达到最大探索轮数
            self.total_pulls = sum(arm_counts)
            if self.max_explore is not None and self.total_pulls >= self.max_explore:
                best_arm = np.argmax(arm_values)
                self.stopped = True
                self.committed_arm = best_arm
                return best_arm
                
            # 计算每个臂的置信区间
            lcb = np.zeros(k)
            ucb = np.zeros(k)
            
            for i in range(k):
                if arm_counts[i] > 0:
                    # 使用Hoeffding不等式计算置信区间
                    bound_width = np.sqrt(np.log(2 * k / delta) / (2 * arm_counts[i]))
                    lcb[i] = arm_values[i] - bound_width
                    ucb[i] = arm_values[i] + bound_width
            
            # 检查是否存在一个臂的LCB高于所有其他臂的UCB
            best_arm = np.argmax(lcb)
            is_confident = True
            
            for i in range(k):
                if i != best_arm and ucb[i] >= lcb[best_arm]:
                    is_confident = False
                    break
            
            if is_confident:
                # 找到确定的最优臂，停止探索
                self.stopped = True
                self.committed_arm = best_arm
                return best_arm
            
            # 未找到确定的最优臂，使用UCB策略继续探索
            return np.argmax(ucb)
    
    class BayesianExploration:
        """
        基于贝叶斯后验概率的动态探索策略。
        当某个臂是最优臂的后验概率超过阈值(1-1/N)时停止探索。
        """
        def __init__(self, threshold_factor=1.0, max_explore=None, num_samples=1000):
            self.threshold_factor = threshold_factor  # 阈值系数
            self.max_explore = max_explore  # 最大探索轮数
            self.num_samples = num_samples  # 蒙特卡洛采样数
            self.stopped = False
            self.committed_arm = None
            self.total_pulls = 0
            
        def reset(self):
            """重置策略状态"""
            self.stopped = False
            self.committed_arm = None
            self.total_pulls = 0
        
        def choose_arm(self, experiment):
            """使用Thompson采样选择臂，并检查是否找到最优臂"""
            if self.stopped:
                return self.committed_arm
                
            k = experiment.k
            arm_counts = experiment.get_arm_counts()
            arm_rewards = experiment.arm_cumulative_rewards
            N = experiment.N
            
            # 计算停止阈值: 与N成正比关系
            threshold = 1.0 - 1.0 / (N * self.threshold_factor)
            threshold = min(threshold, 0.9999)  # 确保阈值合理
            
            # 优先选择未被拉动的臂
            zero_indices = np.where(arm_counts == 0)[0]
            if len(zero_indices) > 0:
                return zero_indices[0]
                
            # 检查是否达到最大探索轮数
            self.total_pulls = sum(arm_counts)
            if self.max_explore is not None and self.total_pulls >= self.max_explore:
                # 达到最大探索轮数，使用后验均值选择最佳臂
                posterior_means = np.zeros(k)
                for i in range(k):
                    if arm_counts[i] > 0:
                        alpha = arm_rewards[i] + 1
                        beta = arm_counts[i] - arm_rewards[i] + 1
                        posterior_means[i] = alpha / (alpha + beta)
                    
                best_arm = np.argmax(posterior_means)
                self.stopped = True
                self.committed_arm = best_arm
                return best_arm
            
            # 只有当每个臂都被拉动过，才计算后验概率
            if np.all(arm_counts > 0):
                # 使用Monte Carlo采样估计后验概率
                samples = np.zeros((self.num_samples, k))
                for i in range(k):
                    alpha = arm_rewards[i] + 1
                    beta = arm_counts[i] - arm_rewards[i] + 1
                    samples[:, i] = np.random.beta(alpha, beta, self.num_samples)
                
                # 计算每个臂成为最优臂的概率
                best_counts = np.zeros(k)
                for s in range(self.num_samples):
                    best_arm_idx = np.argmax(samples[s])
                    best_counts[best_arm_idx] += 1
                
                probabilities = best_counts / self.num_samples
                best_arm = np.argmax(probabilities)
                best_prob = probabilities[best_arm]
                
                # 检查是否超过阈值
                if best_prob >= threshold:
                    self.stopped = True
                    self.committed_arm = best_arm
                    return best_arm
            
            # 未停止时，使用Thompson采样
            samples = np.zeros(k)
            for i in range(k):
                if arm_counts[i] == 0:
                    samples[i] = np.random.beta(1, 1)
                else:
                    alpha = arm_rewards[i] + 1
                    beta = arm_counts[i] - arm_rewards[i] + 1
                    samples[i] = np.random.beta(alpha, beta)
            
            return np.argmax(samples)
