# File: strategies.py
import numpy as np
from scipy import stats # 移到文件顶部，用于 LCB
import math # Add math import for log

# --- 实验阶段策略 ---
class CombinedStrategies:
    """
    包含用于多臂赌博机问题“实验阶段”的各种策略。
    这些策略通常结合了探索 (Exploration) 和利用 (Exploitation)。
    """
    @staticmethod
    def epsilon_greedy(experiment, epsilon=0.1):
        """
        ε-贪婪策略: 以 ε 概率随机探索，以 1-ε 概率选择当前最优臂。

        参数:
        experiment: BanditExperiment 对象, 用于获取当前状态 (arm_values)
        epsilon: 探索概率 (0 <= epsilon <= 1)

        返回:
        选择的臂的索引 (0 to k-1)
        """
        if np.random.random() < epsilon:
            # 探索: 随机选择一个臂
            return np.random.randint(experiment.k)
        else:
            # 利用: 选择当前估计值最高的臂
            arm_values = experiment.get_arm_values()
            # 处理所有臂估计值相同的情况 (例如初始状态)
            if len(set(arm_values)) == 1:
                 return np.random.randint(experiment.k)
             # Find indices of all max values
            max_indices = np.where(arm_values == np.max(arm_values))[0]
             # Randomly choose one if there are multiple max values
            return np.random.choice(max_indices)


    @staticmethod
    def ucb(experiment, c=2.0):
        """
        上置信界 (UCB1) 策略: 选择具有最高 UCB 指数的臂。
        UCB 指数 = 经验均值 + 探索奖励项

        参数:
        experiment: BanditExperiment 对象, 用于获取当前状态 (arm_values, arm_counts)
        c: 探索参数，平衡探索与利用 (c > 0)

        返回:
        选择的臂的索引 (0 to k-1)
        """
        t = sum(experiment.get_arm_counts()) + 1 # 当前总轮次 (从 1 开始计数)
        arm_values = experiment.get_arm_values()
        arm_counts = experiment.get_arm_counts()

        # 优先选择从未被选择过的臂
        zero_indices = np.where(arm_counts == 0)[0]
        if len(zero_indices) > 0:
            # 找到第一个未被选择的臂的索引
            return zero_indices[0]

        # 计算 UCB 探索奖励项
        # Ensure t > 0 and arm_counts[i] > 0 before taking log/sqrt
        # Note: zero_indices check handles arm_counts[i] == 0 case
        if t == 1: # Avoid log(1) = 0 in the first round after initial pulls
             exploration_bonus = np.inf
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
        从每个臂的后验分布中采样，选择样本值最大的臂。

        注意: 此实现假设奖励为 Bernoulli (0 或 1) 来更新 Beta 分布。
              如果实际奖励在 [0, 1] 范围内且非二元,
              使用 Beta 分布可能不是最准确的模型。
              可以考虑其他分布 (如高斯) 或奖励转换方法。

        参数:
        experiment: BanditExperiment 对象, 用于获取当前状态 (arm_values, arm_counts)

        返回:
        选择的臂的索引 (0 to k-1)
        """
        samples = np.zeros(experiment.k)
        arm_cumulative_rewards = experiment.arm_cumulative_rewards # Use cumulative rewards directly
        arm_counts = experiment.get_arm_counts()

        for arm in range(experiment.k):
            if arm_counts[arm] == 0:
                # 对于未探索的臂，从先验 Beta(1, 1) 采样
                samples[arm] = np.random.beta(1, 1)
                continue

            # --- 关键假设区域: Bernoulli 奖励 --- 
            # successes = arm_values[arm] * arm_counts[arm] # Original, less robust
            # Use cumulative rewards assuming they are sums of 0/1s
            # This is still an approximation if rewards aren't Bernoulli
            successes = arm_cumulative_rewards[arm]
            failures = arm_counts[arm] - successes
            # ---------------------------------------

            # Beta 分布参数 (alpha=成功+1, beta=失败+1)
            # 确保 alpha, beta >= 1
            alpha = max(1, successes + 1)
            beta = max(1, failures + 1)

            # 从 Beta 后验分布中抽取样本
            samples[arm] = np.random.beta(alpha, beta)

        # 处理多个臂具有相同最高采样值的情况
        max_sample_indices = np.where(samples == np.max(samples))[0]
        return np.random.choice(max_sample_indices)


    @staticmethod
    def softmax(experiment, temperature=0.1):
        """
        Softmax (Boltzmann Exploration) 策略: 根据估计值的指数权重随机选择臂。

        参数:
        experiment: BanditExperiment 对象, 用于获取当前状态 (arm_values)
        temperature: 温度参数 ( > 0)。值越高，选择越随机；值越低，越趋向于贪婪。

        返回:
        选择的臂的索引 (0 to k-1)
        """
        if temperature <= 0:
            raise ValueError("Temperature must be positive.")

        arm_values = experiment.get_arm_values()
        # 防止溢出: 对所有值减去最大值
        stable_values = arm_values - np.max(arm_values)
        # 计算指数权重
        with np.errstate(over='ignore'): # Ignore overflow, will result in inf
            exp_values = np.exp(stable_values / temperature)

        # 处理 inf (如果一个臂远好于其他臂且温度低)
        if np.isinf(exp_values).any():
            inf_indices = np.where(np.isinf(exp_values))[0]
            return np.random.choice(inf_indices) # 在最优臂中随机选

        sum_exp_values = np.sum(exp_values)

        # 计算概率分布
        if sum_exp_values == 0 or np.isnan(sum_exp_values): # Handle all zero or NaN sum
            # 如果所有 exp_values 都为 0 (可能因为 stable_values 极小或温度极大)
            # 或者出现 NaN，则均匀随机选择
             return np.random.randint(experiment.k)
        else:
            probabilities = exp_values / sum_exp_values

        # 根据概率分布随机选择臂
        return np.random.choice(experiment.k, p=probabilities)

    @staticmethod
    def hybrid_ucb_thompson(experiment, weight=0.5, c=2.0):
        """
        UCB 和汤普森采样的混合策略 (示例)。

        参数:
        experiment: BanditExperiment 对象
        weight: UCB 指数的权重 (0 <= weight <= 1)
        c: UCB 探索参数

        返回:
        选择的臂的索引
        """
        if not (0 <= weight <= 1):
            raise ValueError("Weight must be between 0 and 1.")

        # --- 计算 UCB 部分 ---
        t = sum(experiment.get_arm_counts()) + 1
        arm_values = experiment.get_arm_values()
        arm_counts = experiment.get_arm_counts()
        ucb_part = np.zeros(experiment.k)

        # 优先选择未被选择过的臂
        zero_indices = np.where(arm_counts == 0)[0]
        if len(zero_indices) > 0:
             return zero_indices[0] # 选择第一个未探索的

        if t == 1:
             exploration_bonus = np.inf
        else:
             exploration_bonus = c * np.sqrt(np.log(t) / arm_counts)
        ucb_part = arm_values + exploration_bonus

        # --- 计算汤普森采样部分 ---
        ts_samples = np.zeros(experiment.k)
        arm_cumulative_rewards = experiment.arm_cumulative_rewards
        for arm in range(experiment.k):
            if arm_counts[arm] == 0:
                ts_samples[arm] = np.random.beta(1, 1)
                continue
            successes = arm_cumulative_rewards[arm]
            failures = arm_counts[arm] - successes
            alpha = max(1, successes + 1)
            beta = max(1, failures + 1)
            ts_samples[arm] = np.random.beta(alpha, beta)
        # --------------------------

        # --- 混合 --- 
        # 简单的加权平均 (可能需要归一化 UCB 和 TS 值以获得更好的效果)
        # 注意：直接加权 UCB 值和 TS 样本可能不是理论上最合理的方式
        # 但作为示例可以工作。
        combined_score = weight * ucb_part + (1 - weight) * ts_samples

        # 处理多个臂具有相同最高分的情况
        max_score_indices = np.where(combined_score == np.max(combined_score))[0]
        return np.random.choice(max_score_indices)

    @staticmethod
    def adaptive_epsilon_greedy(experiment, base_epsilon=0.1, scaling_factor=0.1):
        """
        自适应 ε-贪婪策略: ε 值随时间衰减。
        epsilon_t = base_epsilon / (1 + t * scaling_factor)

        参数:
        experiment: BanditExperiment 对象
        base_epsilon: 初始探索概率
        scaling_factor: 衰减速率因子

        返回:
        选择的臂的索引
        """
        t = sum(experiment.get_arm_counts()) # 当前总轮次 (从 0 开始)
        current_epsilon = base_epsilon / (1 + t * scaling_factor)

        # 使用计算出的 current_epsilon 调用标准 epsilon_greedy
        return CombinedStrategies.epsilon_greedy(experiment, epsilon=current_epsilon)


    @staticmethod
    def dynamic_confidence_bounds(experiment, delta_factor=1.0, confidence_constant=4, max_explore_rounds=None):
        """
        动态探索策略：基于置信区间，直到找到统计上显著最优的臂。
        探索轮数 T 是动态确定的。

        参数:
        experiment: BanditExperiment 对象 (提供 k, N, 拉臂接口)
        delta_factor: 用于计算目标错误率 delta = delta_factor / N
        confidence_constant: 置信区间计算中的常数 C (e.g., 4)
        max_explore_rounds: 最大探索轮数上限 (防止无限循环)

        返回:
        选择的臂的索引 (在探索结束后确定)
        注意：此函数 *执行* 探索循环，并返回最终选择的臂。
              它不像其他策略那样只返回 *下一步* 要拉的臂。
              因此，它需要直接访问拉臂和更新统计数据的方法。
              这与当前 BanditExperiment 的设计略有不符，需要调整。
              **临时方案：** 此函数将模拟探索过程，并返回 *假定* 的承诺臂。
                         实际的探索需要在 Experiment 类中实现。
                         这里我们返回一个指示性的臂选择，基于模拟。
        """
        k = experiment.k
        N = experiment.N
        if N <= 0:
            # 如果没有承诺阶段，此策略无意义，退回标准 UCB
            print("警告: N=0，动态置信界策略回退到标准 UCB")
            return CombinedStrategies.ucb(experiment)

        delta = delta_factor / N
        if delta <= 0 or delta >= 1:
            delta = 1 / N # 保证 delta 合理

        # --- 模拟探索过程 --- (理想情况下这部分逻辑在 Experiment 类中)
        # 初始化模拟统计
        sim_counts = np.zeros(k, dtype=int)
        sim_sums = np.zeros(k)
        sim_means = np.zeros(k)
        t = 0 # 总探索轮数

        # 初始拉动每个臂一次
        for i in range(k):
            # 假设拉动得到奖励 (这里无法真正拉动，用 0.5 替代)
            reward = 0.5 # 模拟奖励
            sim_counts[i] += 1
            sim_sums[i] += reward
            sim_means[i] = sim_sums[i] / sim_counts[i]
            t += 1

        # 适应性探索循环
        current_round = 0
        while True:
            current_round += 1 # 跟踪循环次数
            if max_explore_rounds is not None and t >= max_explore_rounds:
                print(f"警告: 达到最大探索轮数 {max_explore_rounds}，提前停止置信界探索。")
                break

            # 计算置信界 (只为拉动过的臂)
            ucb = np.full(k, -np.inf)
            lcb = np.full(k, np.inf)
            valid_indices = np.where(sim_counts > 0)[0]

            if len(valid_indices) == 0: # 不应发生，但作为保险
                break

            # 避免 log(0) 或除以 0
            safe_counts = sim_counts[valid_indices]
            log_term = np.log(confidence_constant * max(1, t) / delta)
            epsilon = np.sqrt(log_term / (2 * safe_counts))

            ucb[valid_indices] = sim_means[valid_indices] + epsilon
            lcb[valid_indices] = sim_means[valid_indices] - epsilon

            # 检查停止条件
            best_empirical_arm = valid_indices[np.argmax(sim_means[valid_indices])]
            lcb_best = lcb[best_empirical_arm]
            max_ucb_others = -np.inf
            for j in valid_indices:
                if j != best_empirical_arm:
                    max_ucb_others = max(max_ucb_others, ucb[j])

            if lcb_best > max_ucb_others:
                # 找到显著最优臂，停止探索
                break

            # 选择要拉动的臂 (LUCB 风格)
            challenger_arm = -1
            max_ucb_val = -np.inf
            for j in valid_indices:
                if j != best_empirical_arm:
                    if ucb[j] > max_ucb_val:
                        max_ucb_val = ucb[j]
                        challenger_arm = j

            arms_to_pull = [best_empirical_arm]
            if challenger_arm != -1:
                arms_to_pull.append(challenger_arm)
            else: # 如果只有一个有效臂被拉过
                 pass # 只拉最优的

            # 模拟拉动并更新
            for arm_idx in arms_to_pull:
                reward = 0.5 # 模拟奖励
                sim_counts[arm_idx] += 1
                sim_sums[arm_idx] += reward
                sim_means[arm_idx] = sim_sums[arm_idx] / sim_counts[arm_idx]
                t += 1

        # --- 探索结束 --- (模拟)
        # 返回最终经验最优臂作为承诺臂
        final_means = sim_means
        # 处理从未被拉动过的臂 (理论上不应发生，除非 K=1)
        if np.all(sim_counts == 0):
             return np.random.randint(k)

        valid_final_indices = np.where(sim_counts > 0)[0]
        best_arm_final = valid_final_indices[np.argmax(final_means[valid_final_indices])]
        # 处理平局
        max_indices = valid_final_indices[np.where(final_means[valid_final_indices] == np.max(final_means[valid_final_indices]))[0]]
        return np.random.choice(max_indices)


    @staticmethod
    def dynamic_bayesian_probability(experiment, delta_factor=1.0, prior_alpha=1, prior_beta=1, posterior_samples=1000, max_explore_rounds=None):
        """
        动态探索策略：基于贝叶斯后验概率，使用汤普森采样，直到最优臂的后验概率足够高。
        探索轮数 T 是动态确定的。

        参数:
        experiment: BanditExperiment 对象 (提供 k, N, 拉臂接口)
        delta_factor: 用于计算目标错误率 delta = delta_factor / N
        prior_alpha, prior_beta: Beta 先验分布的参数
        posterior_samples: 用于估计后验概率的采样次数
        max_explore_rounds: 最大探索轮数上限

        返回:
        选择的臂的索引 (在探索结束后确定)
        注意：与 dynamic_confidence_bounds 类似，此函数模拟探索过程。
              实际实现需要整合到 Experiment 类中。
        """
        k = experiment.k
        N = experiment.N
        if N <= 0:
            print("警告: N=0，动态贝叶斯策略回退到标准汤普森采样")
            return CombinedStrategies.thompson_sampling(experiment)

        delta = delta_factor / N
        if delta <= 0 or delta >= 1:
            delta = 1 / N

        # --- 模拟探索过程 --- (理想情况下在 Experiment 类中)
        # 初始化模拟 Beta 分布参数
        sim_alpha = np.full(k, prior_alpha)
        sim_beta = np.full(k, prior_beta)
        t = 0 # 总探索轮数

        # 适应性探索循环 (Thompson Sampling)
        while True:
            if max_explore_rounds is not None and t >= max_explore_rounds:
                print(f"警告: 达到最大探索轮数 {max_explore_rounds}，提前停止贝叶斯探索。")
                break

            # 1. 从后验采样
            samples = np.random.beta(sim_alpha, sim_beta)

            # 2. 选择臂
            chosen_arm = np.argmax(samples)
            # 处理平局
            max_indices = np.where(samples == np.max(samples))[0]
            if len(max_indices) > 1:
                chosen_arm = np.random.choice(max_indices)

            # 3. 模拟拉动并更新后验 (假设 Bernoulli 奖励)
            # 假设拉动得到奖励 (这里无法真正拉动，随机模拟 0 或 1)
            reward = np.random.randint(0, 2) # 模拟 Bernoulli 奖励
            if reward == 1:
                sim_alpha[chosen_arm] += 1
            else:
                sim_beta[chosen_arm] += 1
            t += 1

            # 4. 检查停止条件 (计算密集型)
            # 估计每个臂是最优的后验概率 P(arm i is best | data)
            prob_best = np.zeros(k)
            # 通过采样估计
            posterior_draws = np.random.beta(sim_alpha[:, np.newaxis], sim_beta[:, np.newaxis], size=(k, posterior_samples))
            # posterior_draws 的形状是 (k, posterior_samples)
            best_arm_indices = np.argmax(posterior_draws, axis=0) # 找到每次采样中的最优臂索引
            # 计算每个臂成为最优臂的次数
            counts = np.bincount(best_arm_indices, minlength=k)
            prob_best = counts / posterior_samples

            # 找到概率最高的臂及其概率
            best_prob_arm = np.argmax(prob_best)
            max_prob = prob_best[best_prob_arm]

            if max_prob > (1 - delta):
                # 找到足够确信的最优臂，停止探索
                break

        # --- 探索结束 --- (模拟)
        # 返回具有最高后验概率的臂作为承诺臂
        # (或者可以选择后验均值最高的臂)
        final_prob_best = np.zeros(k)
        if t > 0: # 确保至少进行了一轮探索
            final_posterior_draws = np.random.beta(sim_alpha[:, np.newaxis], sim_beta[:, np.newaxis], size=(k, posterior_samples))
            final_best_arm_indices = np.argmax(final_posterior_draws, axis=0)
            final_counts = np.bincount(final_best_arm_indices, minlength=k)
            final_prob_best = final_counts / posterior_samples
        else: # 如果从未探索 (例如 max_explore_rounds=0)
            final_prob_best = np.ones(k) / k # 均匀概率

        best_arm_final = np.argmax(final_prob_best)
        # 处理平局
        max_indices_final = np.where(final_prob_best == np.max(final_prob_best))[0]
        return np.random.choice(max_indices_final)


# --- 承诺阶段策略 ---
class CommitmentStrategies:
    """
    包含用于多臂赌博机问题“承诺阶段”的策略。
    这些策略基于“实验阶段”收集到的信息来选择一个单一的臂进行承诺。
    """
    @staticmethod
    def best_empirical(experiment):
        """
        选择实验阶段估计平均奖励最高的臂。

        参数:
        experiment: BanditExperiment 对象, 用于获取最终状态 (arm_values)

        返回:
        选择的臂的索引 (0 to k-1)
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

        参数:
        experiment: BanditExperiment 对象, 用于获取最终状态 (arm_counts)

        返回:
        选择的臂的索引 (0 to k-1)
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

        参数:
        experiment: BanditExperiment 对象
        confidence_level: 置信水平 (例如 0.95 代表 95% 置信度)

        返回:
        选择的臂的索引
        """
        arm_values = experiment.get_arm_values()
        arm_counts = experiment.get_arm_counts()
        k = experiment.k
        lcb = np.full(k, -np.inf) # Initialize LCBs to negative infinity

        # 如果没有臂被拉动 (T=0)，随机选择
        if np.all(arm_counts == 0):
            return np.random.randint(k)

        alpha = 1.0 - confidence_level # Significance level

        for i in range(k):
            if arm_counts[i] > 0:
                # 使用 t 分布计算置信区间 (如果假设正态分布)
                # 或者使用 Hoeffding 不等式 (更通用，无需分布假设)

                # Hoeffding LCB: mu_hat - sqrt(log(1/alpha) / (2 * n))
                # Note: This is a one-sided bound adjustment
                bound_width = np.sqrt(np.log(1 / alpha) / (2 * arm_counts[i]))
                lcb[i] = arm_values[i] - bound_width
            # else: lcb remains -inf

        # 处理 LCB 相同的情况
        max_lcb_indices = np.where(lcb == np.max(lcb))[0]
        return np.random.choice(max_lcb_indices)

# --- Helper function (if needed) ---
# def calculate_confidence_bounds(...): ...


class DynamicConfidenceBoundsStrategy:
    """
    基于置信区间的动态策略 (PAC-style)。
    在每一步选择臂以缩小置信区间，直到满足停止条件。
    停止条件：一个臂的 LCB 高于所有其他臂的 UCB。
    """
    def __init__(self, delta=0.05):
        """
        初始化策略。
        参数:
        delta: 置信水平参数 (0 < delta < 1)。停止条件基于 1 - delta 置信度。
        """
        if not (0 < delta < 1):
            raise ValueError("Delta must be between 0 and 1.")
        self.delta = delta
        self.stopped = False
        self.committed_arm = None

    def _calculate_bounds(self, experiment):
        """计算当前所有臂的 LCB 和 UCB。"""
        k = experiment.k
        t = sum(experiment.get_arm_counts()) + 1 # Total pulls so far + current one
        arm_values = experiment.get_arm_values()
        arm_counts = experiment.get_arm_counts()
        lcb = np.full(k, -np.inf)
        ucb = np.full(k, np.inf)

        # Avoid log(0) or division by zero
        # Use a small delta adjustment specific to the bounds calculation
        # Hoeffding bound: mu_hat +/- sqrt(log(2/delta) / (2 * n))
        # Or Chernoff bound: mu_hat +/- sqrt(log(c*t^p / delta) / n)
        # Using a simpler UCB1-like bound for calculation ease:
        # bound_width = sqrt(log(t) / n_i) - adjust constant as needed

        for i in range(k):
            if arm_counts[i] > 0:
                # Simplified bound width calculation (similar to UCB but for two-sided interval)
                # A more rigorous bound might use log(k * t^2 / delta) or similar
                bound_width = math.sqrt(2 * math.log(t) / arm_counts[i]) # Example width
                # Alternative using confidence parameter delta directly (Hoeffding-like)
                # bound_width = math.sqrt(math.log(2 / (self.delta / k)) / (2 * arm_counts[i]))

                lcb[i] = arm_values[i] - bound_width
                ucb[i] = arm_values[i] + bound_width
            # else: bounds remain +/- inf

        return lcb, ucb

    def choose_arm(self, experiment):
        """
        在实验的每一步选择一个臂。
        """
        if self.stopped:
            return self.committed_arm

        k = experiment.k
        arm_counts = experiment.get_arm_counts()

        # 优先选择未被选择过的臂
        zero_indices = np.where(arm_counts == 0)[0]
        if len(zero_indices) > 0:
            return zero_indices[0]

        # 计算置信界限
        lcb, ucb = self._calculate_bounds(experiment)

        # 检查停止条件
        best_lcb_arm = np.argmax(lcb)
        is_stopped = True
        for j in range(k):
            if j != best_lcb_arm:
                if lcb[best_lcb_arm] <= ucb[j]:
                    is_stopped = False
                    break

        if is_stopped:
            self.stopped = True
            self.committed_arm = best_lcb_arm
            # print(f"[DynamicConf] Stopping at round {sum(arm_counts)}. Commit to arm {self.committed_arm}")
            return self.committed_arm
        else:
            # 如果未停止，选择哪个臂？
            # 策略 1: 拉动当前 LCB 最高的臂 (利用)
            # return best_lcb_arm
            # 策略 2: 拉动当前 UCB 最高的臂 (探索/乐观)
            # return np.argmax(ucb)
            # 策略 3: 拉动边界最不确定的臂 (e.g., UCB - LCB 最大)
            bound_diff = ucb - lcb
            # Handle potential inf/-inf cases if some arms weren't pulled
            valid_indices = np.where(arm_counts > 0)[0]
            if len(valid_indices) == k:
                 return np.argmax(bound_diff)
            else:
                 # If some arms not pulled, prioritize pulling them
                 # This case is handled by the zero_indices check earlier
                 # Fallback: pull the one with max UCB among pulled arms
                 ucb_pulled = ucb[valid_indices]
                 argmax_ucb_pulled = np.argmax(ucb_pulled)
                 return valid_indices[argmax_ucb_pulled]


class DynamicBayesianStrategy:
    """
    基于贝叶斯后验概率的动态策略。
    使用 Thompson Sampling 选择臂，直到一个臂的后验最优概率超过阈值。
    假设 Bernoulli 奖励和 Beta 先验/后验。
    """
    def __init__(self, delta=0.05, posterior_samples=1000):
        """
        初始化策略。
        参数:
        delta: 停止阈值参数 (0 < delta < 1)。当 P(arm_i is best) > 1 - delta 时停止。
        posterior_samples: 用于估计最优概率的后验样本数量。
        """
        if not (0 < delta < 1):
            raise ValueError("Delta must be between 0 and 1.")
        self.delta = delta
        self.posterior_samples = posterior_samples
        self.stopped = False
        self.committed_arm = None
        # 内部状态: Beta 分布参数 (alpha, beta) for each arm
        # Initialize with prior Beta(1, 1)
        self.alphas = None
        self.betas = None

    def _update_posteriors(self, experiment):
        """根据实验数据更新内部 Beta 后验参数。"""
        k = experiment.k
        if self.alphas is None: # Initialize on first call
            self.alphas = np.ones(k)
            self.betas = np.ones(k)

        # 获取自上次更新以来的新数据 (这里简化：直接用 experiment 的累计数据)
        # 注意：这假设策略对象在每次运行中被重用或正确重置。
        # 一个更健壮的方法是跟踪上次更新时的 counts/rewards。
        # 为了简单起见，我们每次都从头计算后验参数。
        arm_cumulative_rewards = experiment.arm_cumulative_rewards
        arm_counts = experiment.get_arm_counts()

        for arm in range(k):
            if arm_counts[arm] > 0:
                successes = arm_cumulative_rewards[arm] # Assumes Bernoulli rewards sum
                failures = arm_counts[arm] - successes
                self.alphas[arm] = max(1, successes + 1) # Prior is Beta(1,1)
                self.betas[arm] = max(1, failures + 1)
            else:
                self.alphas[arm] = 1
                self.betas[arm] = 1

    def _check_stopping_condition(self, experiment):
        """通过从后验采样来检查停止条件。"""
        k = experiment.k
        samples = np.zeros((self.posterior_samples, k))

        for arm in range(k):
            samples[:, arm] = np.random.beta(self.alphas[arm], self.betas[arm], size=self.posterior_samples)

        # 计算每个臂是样本中最大值的次数
        best_arm_counts = np.zeros(k)
        best_indices = np.argmax(samples, axis=1)
        for i in range(k):
            best_arm_counts[i] = np.sum(best_indices == i)

        # 计算每个臂是最优的后验概率
        posterior_prob_best = best_arm_counts / self.posterior_samples

        # 检查是否有任何臂的概率超过阈值
        best_prob_arm = np.argmax(posterior_prob_best)
        if posterior_prob_best[best_prob_arm] > (1 - self.delta):
            return True, best_prob_arm
        else:
            return False, None

    def choose_arm(self, experiment):
        """
        在实验的每一步选择一个臂。
        """
        if self.stopped:
            return self.committed_arm

        k = experiment.k
        arm_counts = experiment.get_arm_counts()

        # 优先选择未被选择过的臂 (确保 Beta 参数至少为 1,1)
        zero_indices = np.where(arm_counts == 0)[0]
        if len(zero_indices) > 0:
            # 更新内部状态以反映先验，即使没有拉动
            if self.alphas is None: self._update_posteriors(experiment)
            return zero_indices[0]

        # 更新后验分布
        self._update_posteriors(experiment)

        # 检查停止条件
        should_stop, commit_arm = self._check_stopping_condition(experiment)
        if should_stop:
            self.stopped = True
            self.committed_arm = commit_arm
            # print(f"[DynamicBayes] Stopping at round {sum(arm_counts)}. Commit to arm {self.committed_arm}")
            return self.committed_arm
        else:
            # 如果未停止，使用 Thompson Sampling 选择下一个臂
            # (从当前更新的后验中采样一次)
            ts_samples = np.zeros(k)
            for arm in range(k):
                ts_samples[arm] = np.random.beta(self.alphas[arm], self.betas[arm])

            max_sample_indices = np.where(ts_samples == np.max(ts_samples))[0]
            return np.random.choice(max_sample_indices)

    def reset(self): # Add reset method if the object is reused across runs
        """重置内部状态。"""
        self.stopped = False
        self.committed_arm = None
        self.alphas = None
        self.betas = None


# --- 承诺阶段策略 --- 
class CommitmentStrategies:
    """
    包含用于多臂赌博机问题“承诺阶段”的策略。
    这些策略基于“实验阶段”收集到的信息来选择一个单一的臂进行承诺。
    """
    @staticmethod
    def best_empirical(experiment):
        """
        选择实验阶段结束时，经验平均奖励最高的臂。

        参数:
        experiment: BanditExperiment 对象, 用于获取最终状态 (arm_values)

        返回:
        选择的臂的索引 (0 to k-1)
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

        参数:
        experiment: BanditExperiment 对象, 用于获取最终状态 (arm_counts)

        返回:
        选择的臂的索引 (0 to k-1)
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

        参数:
        experiment: BanditExperiment 对象
        confidence_level: 置信水平 (例如 0.95 代表 95% 置信度)

        返回:
        选择的臂的索引
        """
        arm_values = experiment.get_arm_values()
        arm_counts = experiment.get_arm_counts()
        k = experiment.k
        lcb = np.full(k, -np.inf) # Initialize LCBs to negative infinity

        # 如果没有臂被拉动 (T=0)，随机选择
        if np.all(arm_counts == 0):
            return np.random.randint(k)

        alpha = 1.0 - confidence_level # Significance level

        for i in range(k):
            if arm_counts[i] > 0:
                # 使用 t 分布计算置信区间 (如果假设正态分布)
                # 或者使用 Hoeffding 不等式 (更通用，无需分布假设)

                # Hoeffding LCB: mu_hat - sqrt(log(1/alpha) / (2 * n))
                # Note: This is a one-sided bound adjustment
                bound_width = np.sqrt(np.log(1 / alpha) / (2 * arm_counts[i]))
                lcb[i] = arm_values[i] - bound_width
            # else: lcb remains -inf

        # 处理 LCB 相同的情况
        max_lcb_indices = np.where(lcb == np.max(lcb))[0]
        return np.random.choice(max_lcb_indices)

# --- Helper function (if needed) ---
# def calculate_confidence_bounds(...): ...