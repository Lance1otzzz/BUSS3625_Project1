# File: visualization.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib as mpl
import platform

# 设置可视化样式
sns.set_style('whitegrid')
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14

# 配置支持中文字符的字体
def configure_chinese_font():
    """配置支持中文字符的字体"""
    system = platform.system()
    if system == 'Darwin':  # macOS
        # macOS 常见的支持中文的字体
        chinese_fonts = ['Arial Unicode MS', 'PingFang SC', 'Heiti SC', 'STHeiti', 'Apple LiGothic']
    elif system == 'Windows':
        chinese_fonts = ['Microsoft YaHei', 'SimHei', 'SimSun', 'NSimSun', 'FangSong']
    else:  # Linux 或其他
        chinese_fonts = ['WenQuanYi Micro Hei', 'Droid Sans Fallback', 'Noto Sans CJK SC']
    
    # 尝试设置字体
    for font in chinese_fonts:
        try:
            mpl.rcParams['font.family'] = [font, 'sans-serif']
            # 测试字体是否可用
            mpl.font_manager.findfont(mpl.font_manager.FontProperties(family=font))
            print(f"Using font: {font} for CJK characters")
            return True
        except Exception:
            continue
    
    # 如果没有找到合适的字体，使用默认字体并发出警告
    print("Warning: No suitable font for Chinese characters found. Using default font.")
    return False

# 调用字体配置函数
configure_chinese_font()

def visualize_single_run_details(experiment, strategy_name="Unknown Strategy", commitment_name="Unknown Commitment", filename_prefix="single_run"):
    """
    可视化单次实验的详细结果。
    """
    experiment_arms, experiment_rewards = experiment.get_experiment_data()
    commitment_arm, commitment_rewards = experiment.get_commitment_data()
    k = experiment.k
    T = experiment.T
    N = experiment.N
    true_means = experiment.bandit.get_true_means()
    arm_counts = experiment.get_arm_counts()
    arm_values = experiment.get_arm_values()
    optimal_arm_idx = experiment.bandit.get_optimal_arm()

    fig, axes = plt.subplots(2, 2, figsize=(16, 13))
    ax1, ax2, ax3, ax4 = axes.flatten()
    fig.suptitle(f'Single Run: Exp="{strategy_name}", Commit="{commitment_name}"\n(T={T}, N={N}, k={k})', fontsize=18)

    # 1. 实验阶段：臂选择次数
    sns.barplot(x=list(range(k)), y=arm_counts, ax=ax1, palette='viridis')
    ax1.set_title(f'Experiment Phase (T={T}): Arm Pull Counts')
    ax1.set_xlabel('Arm Index')
    ax1.set_ylabel('Number of Pulls')
    if commitment_arm is not None:
        ax1.get_xticklabels()[commitment_arm].set_weight('bold')
        ax1.get_xticklabels()[commitment_arm].set_color('blue')
    ax1.get_xticklabels()[optimal_arm_idx].set_color('red')

    # 2. 估计vs真实均值
    x = np.arange(k)
    width = 0.35
    ax2.bar(x - width/2, arm_values, width, label=f'Estimated Mean (at T={T})', color='skyblue')
    ax2.bar(x + width/2, true_means, width, label='True Mean', color='salmon', alpha=0.8)
    ax2.axvline(optimal_arm_idx + width/2, color='red', linestyle='--', linewidth=2, label=f'Optimal Arm ({optimal_arm_idx})')
    if commitment_arm is not None:
        ax2.axvline(commitment_arm - width/2, color='blue', linestyle=':', linewidth=2, label=f'Committed Arm ({commitment_arm})')
    ax2.set_title('Mean Rewards: Estimated vs. True')
    ax2.set_xlabel('Arm Index')
    ax2.set_ylabel('Mean Reward')
    ax2.set_xticks(x)
    ax2.legend(fontsize=10)
    if commitment_arm is not None:
        ax2.get_xticklabels()[commitment_arm].set_weight('bold')
        ax2.get_xticklabels()[commitment_arm].set_color('blue')
    ax2.get_xticklabels()[optimal_arm_idx].set_color('red')

    # 3. 实验阶段：累积奖励
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

    # 4. 承诺阶段：累积奖励
    if N > 0 and commitment_rewards and commitment_arm is not None:
        cumulative_rewards_com = np.cumsum(commitment_rewards)
        sns.lineplot(x=range(T + 1, T + N + 1), y=cumulative_rewards_com, ax=ax4, color='purple')
        ax4.set_title(f'Commitment Phase (N={N}, Arm={commitment_arm}): Cumulative Reward')
        ax4.set_xlabel(f'Round ({T+1} to {T+N})')
        ax4.set_ylabel('Cumulative Reward during Commitment')
        # 添加最优奖励线
        optimal_commit_reward = experiment.bandit.get_true_means()[commitment_arm] * np.arange(1, N + 1)
        ax4.plot(range(T + 1, T + N + 1), optimal_commit_reward, linestyle=':', color='gray', label=f'Expected Reward if Arm {commitment_arm} Pulled')
        ax4.legend(fontsize=9)
        ax4.grid(True, linestyle='--', alpha=0.6)
    else:
        ax4.text(0.5, 0.5, 'No Commitment Phase (N=0)', horizontalalignment='center', verticalalignment='center', transform=ax4.transAxes)
        ax4.set_title('Commitment Phase (N=0)')

    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    clean_strategy_name = "".join(c if c.isalnum() else "_" for c in strategy_name)
    clean_commit_name = "".join(c if c.isalnum() else "_" for c in commitment_name)
    filename = f'{filename_prefix}_{clean_strategy_name}_{clean_commit_name}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  Single run details saved to: {filename}")
    plt.close(fig)

def visualize_strategy_comparison(results_dict, T, N, k, num_runs, filename_prefix="comparison"):
    """
    可视化不同策略的最终平均总后悔值对比。
    """
    if not results_dict:
        print("Warning: No results found in comparison dictionary. Skipping plot.")
        return

    # 转换为DataFrame便于排序/绘图
    df = pd.DataFrame.from_dict(results_dict, orient='index')
    df = df.sort_values(by='mean')

    # 创建图形
    n_strategies = len(df)
    fig, ax = plt.subplots(figsize=(max(12, n_strategies * 0.7), 8))

    # 条形图
    colors = sns.color_palette("viridis", n_strategies)
    bars = ax.bar(df.index, df['mean'], yerr=df['std'], capsize=5, color=colors)

    ax.set_ylabel('Average Total Regret (Lower is Better)')
    ax.set_xlabel('Strategy Combination (Experiment Policy + Commitment Policy)')
    ax.set_title(f'Strategy Comparison: Average Total Regret\n(T={T}, N={N}, k={k}, Runs={num_runs})')
    ax.tick_params(axis='x', rotation=45)

    # 添加数值标签
    ax.bar_label(bars, fmt='%.2f', padding=3, fontsize=10)

    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    filename = f'{filename_prefix}_T{T}_N{N}_k{k}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  Strategy comparison saved to: {filename}")
    plt.close(fig)

def visualize_average_regret_over_time(regret_data, T, N, k, num_runs, filename_prefix="avg_regret_time"):
    """
    可视化不同策略的平均累积后悔值随时间变化。
    """
    if not regret_data:
        print("Warning: No regret time-series data provided. Skipping plot.")
        return

    plt.figure(figsize=(14, 8))
    total_rounds = T + N
    rounds = np.arange(1, total_rounds + 1)

    # 定义颜色
    colors = sns.color_palette("husl", len(regret_data))
    color_map = {name: color for name, color in zip(regret_data.keys(), colors)}

    # 根据最终平均后悔值排序策略
    final_mean_regrets = {name: np.mean(data[:, -1]) for name, data in regret_data.items()}
    sorted_names = sorted(regret_data.keys(), key=lambda name: final_mean_regrets[name])

    for name in sorted_names:
        data = regret_data[name] # Shape: (num_runs, T+N)
        mean_regret = np.mean(data, axis=0)
        std_regret = np.std(data, axis=0)

        # 绘制平均后悔值线
        plt.plot(rounds, mean_regret, label=f"{name} (Final Avg: {mean_regret[-1]:.2f})", color=color_map[name], linewidth=2)

    # 添加分隔实验和承诺阶段的垂直线
    if T > 0 and N > 0:
        plt.axvline(x=T, color='gray', linestyle=':', linewidth=2, label=f'End of Experiment Phase (T={T})')

    plt.title(f'Average Cumulative Regret Over Time\n(T={T}, N={N}, k={k}, Runs={num_runs})')
    plt.xlabel('Round Number (1 to T+N)')
    plt.ylabel('Average Cumulative Regret')
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlim(1, total_rounds)
    plt.ylim(bottom=0)

    plt.tight_layout()
    filename = f'{filename_prefix}_T{T}_N{N}_k{k}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  Average regret over time saved to: {filename}")
    plt.close()

def visualize_tn_relation(results_dict, k, num_runs, filename_prefix="tn_relation"):
    """
    可视化T和N关系对后悔值的影响
    
    Args:
        results_dict: 字典，键为策略名，值为(T,N,后悔值)列表
        k: 臂数
        num_runs: 每个配置的运行次数
        filename_prefix: 输出文件前缀
    """
    # 使用支持中文的字体
    plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
    # 为每个策略创建单独的图表
    for strategy_name, results in results_dict.items():
        plt.figure(figsize=(10, 8))
        
        # 提取数据
        t_values = [t for t, _, _ in results]
        n_values = [n for _, n, _ in results]
        regrets = [r for _, _, r in results]
        
        # 创建散点图，颜色代表后悔值
        scatter = plt.scatter(t_values, n_values, c=regrets, cmap='viridis', 
                              s=100, alpha=0.7)
        
        # 添加颜色条和标签
        plt.colorbar(scatter, label='Average Regret')
        for i, (t, n, r) in enumerate(results):
            plt.annotate(f"{r:.1f}", (t, n), xytext=(5, 5), textcoords='offset points')
        
        # 绘制T/N比率线（可选）
        t_max = max(t_values)
        n_max = max(n_values)
        for ratio in [0.5, 1, 2]:
            t_line = np.linspace(0, min(t_max, ratio * n_max), 100)
            n_line = t_line / ratio
            plt.plot(t_line, n_line, 'r--', linewidth=1, alpha=0.5)
            plt.text(t_max/2, t_max/(2*ratio), f'T/N={ratio}', 
                     color='red', alpha=0.7, ha='center', va='center')
        
        # 找到最小后悔值点
        min_idx = np.argmin(regrets)
        min_t, min_n, min_regret = results[min_idx]
        min_ratio = min_t / min_n if min_n > 0 else float('inf')
        
        plt.scatter([min_t], [min_n], color='red', s=150, marker='*')
        plt.annotate(f'Min Regret: {min_regret:.2f}\n(T={min_t}, N={min_n}, T/N={min_ratio:.2f})',
                     (min_t, min_n), xytext=(15, 15), textcoords='offset points',
                     bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                     arrowprops=dict(arrowstyle='->'))
        
        plt.xlabel('Experiment Phase Length (T)')
        plt.ylabel('Commitment Phase Length (N)')
        plt.title(f'T/N Relationship Analysis for {strategy_name}\n(k={k}, runs={num_runs})')
        plt.grid(True, linestyle='--', alpha=0.3)
        
        # 保存图像
        clean_strategy = strategy_name.replace(' ', '_').replace('(', '').replace(')', '')
        filename = f'{filename_prefix}_{clean_strategy}_k{k}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  T/N relation plot saved to: {filename}")
def visualize_dynamic_strategies(results, k, num_runs, filename_prefix="dynamic_strategies"):
    """
    可视化动态探索策略与固定策略的比较结果
    
    参数:
        results: 格式为 {(策略类型, 策略名, N): (avg_regret, avg_t, std_regret)}
        k: 臂数
        num_runs: 每个配置的运行次数
        filename_prefix: 输出文件前缀
    """
    # 使用支持中文的字体
    plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
    # 按N分组
    n_values = sorted(list(set(n for _, _, n in results.keys())))
    
    for n in n_values:
        plt.figure(figsize=(12, 8))
        
        # 提取该N值的结果
        n_results = {k: v for k, v in results.items() if k[2] == n}
        
        # 分离固定策略和动态策略
        fixed_results = {k: v for k, v in n_results.items() if k[0] == 'fixed'}
        dynamic_results = {k: v for k, v in n_results.items() if k[0] == 'dynamic'}
        
        # 按策略名分组固定策略结果
        fixed_data = {}
        for (_, name, _), (regret, t, _) in fixed_results.items():
            base_name = name.split(' (T=')[0]
            if base_name not in fixed_data:
                fixed_data[base_name] = []
            fixed_data[base_name].append((t, regret))
        
        # 绘制固定策略曲线
        for base_name, points in fixed_data.items():
            points.sort(key=lambda x: x[0])  # 按T排序
            t_values = [p[0] for p in points]
            regrets = [p[1] for p in points]
            plt.plot(t_values, regrets, 'o-', label=f'{base_name} (固定T)')
        
        # 绘制动态策略点
        markers = ['*', 'P', 'X', 'h', 'D']
        colors = plt.cm.tab10(np.linspace(0, 1, len(dynamic_results)))
        
        for i, ((_, name, _), (regret, avg_t, std_regret)) in enumerate(dynamic_results.items()):
            plt.errorbar([avg_t], [regret], yerr=[std_regret], fmt=markers[i % len(markers)], 
                         markersize=12, color=colors[i], capsize=5,
                         label=f'{name} (平均T={avg_t:.1f})')
        
        plt.xlabel('探索阶段长度 (T)')
        plt.ylabel('平均后悔值')
        plt.title(f'动态探索策略与固定T策略比较\n(N={n}, k={k}, 运行次数={num_runs})')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='best')
        
        plt.tight_layout()
        filename = f'{filename_prefix}_N{n}_k{k}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  动态策略比较图已保存至: {filename}")

def visualize_tn_impact_on_exploration(results, filename_prefix="tn_impact"):
    """
    可视化N值对动态策略探索轮数的影响
    
    参数:
        results: 格式为 {(策略名, N): (avg_regret, avg_t, std_regret)}
        filename_prefix: 输出文件前缀
    """
    # 使用支持中文的字体
    plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
    plt.figure(figsize=(10, 8))
    
    # 按策略分组
    strategy_data = {}
    for (name, n), (_, avg_t, _) in results.items():
        if name not in strategy_data:
            strategy_data[name] = []
        strategy_data[name].append((n, avg_t))
    
    # 绘制每个策略的曲线
    markers = ['o', 's', '^', 'D', 'P']
    colors = plt.cm.tab10(np.linspace(0, 1, len(strategy_data)))
    
    for i, (name, points) in enumerate(strategy_data.items()):
        points.sort(key=lambda x: x[0])  # 按N排序
        n_values = [p[0] for p in points]
        avg_t_values = [p[1] for p in points]
        
        plt.plot(n_values, avg_t_values, marker=markers[i % len(markers)], 
                 linestyle='-', color=colors[i], label=name)
    
    plt.xscale('log')
    plt.xlabel('承诺阶段长度 (N)')
    plt.ylabel('平均探索轮数 (T)')
    plt.title('N值对动态策略探索轮数的影响')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best')
    
    plt.tight_layout()
    filename = f'{filename_prefix}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  N值影响图已保存至: {filename}")
