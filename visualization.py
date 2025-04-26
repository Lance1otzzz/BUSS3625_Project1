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

def visualize_single_run_details(experiment, strategy_name="Unknown Strategy", commitment_name="Unknown Commitment", filename_prefix="dynamic_analysis/single_run"):
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

def visualize_strategy_comparison(results_dict, T, N, k, num_runs, filename_prefix="dynamic_analysis/comparison", dynamic_results=None):
    """
    可视化不同策略的最终平均总后悔值对比。
    
    参数:
        results_dict: 常规策略结果字典
        T: 探索阶段长度
        N: 承诺阶段长度
        k: 臂数
        num_runs: 运行次数
        filename_prefix: 输出文件前缀
        dynamic_results: 动态策略结果，格式为 {(策略类型, 策略名, N): (avg_regret, avg_t, std_regret)}
    """
    if not results_dict and (dynamic_results is None or len(dynamic_results) == 0):
        print("Warning: No results found in comparison dictionary. Skipping plot.")
        return

    # 转换为DataFrame便于排序/绘图
    df = pd.DataFrame.from_dict(results_dict, orient='index')
    
    # 添加动态策略结果（如果有）
    if dynamic_results:
        dynamic_df_data = {}
        for (strategy_type, name, n), (avg_regret, avg_t, std_regret) in dynamic_results.items():
            if strategy_type == 'dynamic' and n == N:  # 只选择匹配当前N值的动态策略
                key = f"{name} (动态T≈{avg_t:.1f})"
                dynamic_df_data[key] = {'mean': avg_regret, 'std': std_regret}
        
        # 合并到主DataFrame
        if dynamic_df_data:
            dynamic_df = pd.DataFrame.from_dict(dynamic_df_data, orient='index')
            df = pd.concat([df, dynamic_df])
    
    # 按平均后悔值排序
    df = df.sort_values(by='mean')

    # 创建图形
    n_strategies = len(df)
    fig, ax = plt.subplots(figsize=(max(12, n_strategies * 0.7), 8))

    # 设置颜色 - 常规策略使用蓝色系，动态策略使用红色系
    regular_count = len(results_dict)
    dynamic_count = n_strategies - regular_count
    colors = []
    if regular_count > 0:
        colors.extend(sns.color_palette("Blues_d", regular_count))
    if dynamic_count > 0:
        colors.extend(sns.color_palette("Reds_d", dynamic_count))

    # 条形图
    bars = ax.bar(df.index, df['mean'], yerr=df['std'], capsize=5, color=colors)

    # 添加标题和标签
    ax.set_ylabel('平均总后悔值', fontsize=14)
    ax.set_xlabel('策略组合', fontsize=14)
    ax.set_title(f'策略比较: 平均总后悔值\n(T={T}, N={N}, k={k}, 运行次数={num_runs})', fontsize=16)
    ax.tick_params(axis='x', rotation=45, labelsize=11)

    # 添加数值标签
    ax.bar_label(bars, fmt='%.2f', padding=3, fontsize=10)

    # 添加图例来区分常规和动态策略
    if dynamic_count > 0 and regular_count > 0:
        import matplotlib.patches as mpatches
        regular_patch = mpatches.Patch(color=colors[0], label='常规策略')
        dynamic_patch = mpatches.Patch(color=colors[-1], label='动态策略')
        ax.legend(handles=[regular_patch, dynamic_patch], loc='upper right')

    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    filename = f'{filename_prefix}_T{T}_N{N}_k{k}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"  策略比较已保存至: {filename}")
    plt.close(fig)

def visualize_average_regret_over_time(regret_data, T, N, k, num_runs, filename_prefix="dynamic_analysis/avg_regret_time"):
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

# 此处移除了visualize_tn_relation函数，因为不再需要TN analysis相关功能
def visualize_dynamic_strategies(results, k, num_runs, filename_prefix="dynamic_analysis/dynamic_strategies"):
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
        # 创建一个具有两个子图的画布：左侧是主要比较图，右侧是比较表格
        fig = plt.figure(figsize=(16, 9))
        gs = fig.add_gridspec(1, 2, width_ratios=[3, 1])  # 网格布局，3:1宽度比
        ax1 = fig.add_subplot(gs[0, 0])  # 左侧图表
        ax2 = fig.add_subplot(gs[0, 1])  # 右侧表格
        
        # 提取该N值的结果
        n_results = {k: v for k, v in results.items() if k[2] == n}
        
        # 分离固定策略和动态策略
        fixed_results = {k: v for k, v in n_results.items() if k[0] == 'fixed'}
        dynamic_results = {k: v for k, v in n_results.items() if k[0] == 'dynamic'}
        
        # 按策略名分组固定策略结果
        fixed_data = {}
        for (_, name, _), (regret, t, std_regret) in fixed_results.items():
            base_name = name.split(' (T=')[0]
            if base_name not in fixed_data:
                fixed_data[base_name] = []
            fixed_data[base_name].append((t, regret, std_regret))
        
        # 绘制固定策略曲线
        for base_name, points in fixed_data.items():
            points.sort(key=lambda x: x[0])  # 按T排序
            t_values = [p[0] for p in points]
            regrets = [p[1] for p in points]
            std_regrets = [p[2] for p in points]
            line, = ax1.plot(t_values, regrets, 'o-', label=f'{base_name} (固定T)', linewidth=2)
            color = line.get_color()
            # 添加标准差区域
            ax1.fill_between(t_values, 
                             [r - s for r, s in zip(regrets, std_regrets)],
                             [r + s for r, s in zip(regrets, std_regrets)],
                             color=color, alpha=0.2)
        
        # 绘制动态策略点
        markers = ['*', 'P', 'X', 'h', 'D']
        dynamic_colors = plt.cm.tab10(np.linspace(0, 1, len(dynamic_results)))
        
        # 为表格准备数据
        table_data = []
        table_colors = []
        
        for i, ((_, name, _), (regret, avg_t, std_regret)) in enumerate(dynamic_results.items()):
            ax1.errorbar([avg_t], [regret], yerr=[std_regret], fmt=markers[i % len(markers)], 
                         markersize=14, color=dynamic_colors[i], capsize=6, linewidth=2,
                         label=f'{name} (平均T={avg_t:.1f})')
            
            # 添加透明度更高的圆形区域来强调动态策略点
            ax1.add_patch(plt.Circle((avg_t, regret), radius=max(std_regret*1.5, 20), 
                                     color=dynamic_colors[i], alpha=0.15))
            
            # 收集表格数据
            table_data.append([name, f"{avg_t:.1f}", f"{regret:.2f}±{std_regret:.2f}"])
            table_colors.append([dynamic_colors[i], dynamic_colors[i], dynamic_colors[i]])
        
        # 设置主图样式
        ax1.set_xlabel('探索阶段长度 (T)', fontsize=14)
        ax1.set_ylabel('平均后悔值', fontsize=14)
        ax1.set_title(f'动态探索策略与固定T策略性能比较\n(N={n}, k={k}, 运行次数={num_runs})', fontsize=16)
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend(loc='best', fontsize=11)
        
        # 寻找x轴和y轴的范围
        all_t_values = [t for _, (_, t, _) in n_results.items()]
        all_regrets = [r for _, (r, _, _) in n_results.items()]
        t_min, t_max = min(all_t_values), max(all_t_values)
        r_min, r_max = min(all_regrets), max(all_regrets)
        
        # 扩展轴范围以便更好地显示
        t_range = t_max - t_min
        r_range = r_max - r_min
        ax1.set_xlim([max(0, t_min - 0.1 * t_range), t_max + 0.1 * t_range])
        ax1.set_ylim([max(0, r_min - 0.1 * r_range), r_max + 0.1 * r_range])
        
        # 创建表格
        if table_data:
            ax2.axis('off')  # 隐藏轴
            table = ax2.table(
                cellText=table_data,
                colLabels=['动态策略', '平均T', '平均后悔值±标准差'],
                cellColours=[[(0.95, 0.95, 0.95) for _ in range(3)] for _ in range(len(table_data))],  # 浅灰色背景
                colColours=[(0.8, 0.8, 0.8) for _ in range(3)],  # 表头颜色
                cellLoc='center',
                loc='center'
            )
            table.auto_set_font_size(False)
            table.set_fontsize(11)
            table.scale(1, 1.5)  # 调整表格高度
            
            # 为表格添加标题
            ax2.text(0.5, 0.95, '动态策略性能统计', 
                     horizontalalignment='center',
                     verticalalignment='center',
                     transform=ax2.transAxes,
                     fontsize=14,
                     fontweight='bold')
        
        plt.tight_layout()
        filename = f'{filename_prefix}_N{n}_k{k}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  动态策略比较图已保存至: {filename}")
        
        # 额外创建一个简化版的图表，仅关注动态策略
        if dynamic_results:
            plt.figure(figsize=(12, 8))
            
            # 按后悔值排序动态策略
            sorted_dynamic = sorted(dynamic_results.items(), key=lambda x: x[1][0])
            names = [name for (_, name, _), _ in sorted_dynamic]
            regrets = [regret for _, (regret, _, _) in sorted_dynamic]
            avg_ts = [avg_t for _, (_, avg_t, _) in sorted_dynamic]
            std_regrets = [std for _, (_, _, std) in sorted_dynamic]
            
            # 创建条形图
            bars = plt.bar(names, regrets, yerr=std_regrets, capsize=8, 
                          color=plt.cm.viridis(np.linspace(0, 0.8, len(names))))
            
            # 在每个条形上标注平均T值
            for i, (bar, avg_t) in enumerate(zip(bars, avg_ts)):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std_regrets[i] + 0.5,
                        f'T={avg_t:.1f}', ha='center', va='bottom', fontsize=11)
            
            plt.xlabel('动态探索策略', fontsize=14)
            plt.ylabel('平均后悔值', fontsize=14)
            plt.title(f'动态策略性能比较 (按后悔值排序)\n(N={n}, k={k}, 运行次数={num_runs})', fontsize=16)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.xticks(rotation=20, ha='right')  # 旋转策略名称以防止重叠
            
            plt.tight_layout()
            filename = f'{filename_prefix}_dynamic_only_N{n}_k{k}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  动态策略排行图已保存至: {filename}")

def visualize_tn_impact_on_exploration(results, filename_prefix="dynamic_analysis/tn_impact"):
    """
    可视化N值对动态策略探索轮数的影响
    
    参数:
        results: 格式为 {(策略名, N): (avg_regret, avg_t, std_regret)}
        filename_prefix: 输出文件前缀
    """
    # 使用支持中文的字体
    plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
    
    # 首先创建主曲线图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), gridspec_kw={'width_ratios': [2, 1]})
    
    # 按策略分组
    strategy_data = {}
    regret_data = {}
    for (name, n), (avg_regret, avg_t, std_regret) in results.items():
        if name not in strategy_data:
            strategy_data[name] = []
            regret_data[name] = []
        strategy_data[name].append((n, avg_t, std_regret))
        regret_data[name].append((n, avg_regret))
    
    # 绘制每个策略的T-N曲线
    markers = ['o', 's', '^', 'D', 'P']
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(strategy_data)))
    
    # 添加比例参考线
    n_min, n_max = float('inf'), 0
    t_min, t_max = float('inf'), 0
    
    # 绘制T-N关系曲线
    for i, (name, points) in enumerate(strategy_data.items()):
        points.sort(key=lambda x: x[0])  # 按N排序
        n_values = [p[0] for p in points]
        avg_t_values = [p[1] for p in points]
        std_t_values = [p[2] for p in points]
        
        color = colors[i]
        line = ax1.plot(n_values, avg_t_values, marker=markers[i % len(markers)], 
                 linestyle='-', color=color, linewidth=2, label=name)
        
        # 添加误差范围
        for j, (n, t, std) in enumerate(points):
            ax1.fill_between([n], [t-std], [t+std], color=color, alpha=0.2)
        
        # 在数据点上标注T/N比率
        for j, (n, t, _) in enumerate(points):
            if j % 2 == 0:  # 每隔一个点标注，避免过于拥挤
                ratio = t / n
                ax1.annotate(f'T/N={ratio:.2f}', (n, t), 
                          xytext=(0, 10), textcoords='offset points',
                          ha='center', fontsize=9, color=color)
        
        # 更新最大最小值
        n_min = min(n_min, min(n_values))
        n_max = max(n_max, max(n_values))
        t_min = min(t_min, min(avg_t_values))
        t_max = max(t_max, max(avg_t_values))
    
    # 添加T=N参考线和其他比例线
    n_range = np.linspace(n_min*0.8, n_max*1.2, 100)
    for ratio, style, alpha, label in zip(
        [0.5, 1.0, 2.0], 
        [':', '--', '-.'], 
        [0.5, 0.7, 0.5],
        ['T=0.5N', 'T=N', 'T=2N']):
        ax1.plot(n_range, ratio * n_range, style, color='gray', 
                alpha=alpha, label=label)
    
    # 设置第一个子图的属性
    ax1.set_xscale('log')
    ax1.set_xlabel('承诺阶段长度 (N)', fontsize=14)
    ax1.set_ylabel('平均探索轮数 (T)', fontsize=14)
    ax1.set_title('N值对动态策略探索轮数的影响', fontsize=16)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(loc='best', fontsize=11)
    
    # 绘制有效性分析图 - 探索效率(N/T)与后悔值的关系
    for i, (name, points) in enumerate(strategy_data.items()):
        # 获取对应的后悔值数据
        regret_points = regret_data[name]
        regret_points.sort(key=lambda x: x[0])  # 按N排序
        
        # 计算每个点的N/T比率作为探索效率
        efficiencies = []
        ns = []
        regrets = []
        
        for (n, t, _), (n2, r) in zip(points, regret_points):
            if n != n2:
                continue  # 确保N值匹配
            if t > 0:  # 避免除以零
                efficiency = n / t  # 高比率表示更高的探索效率
                efficiencies.append(efficiency)
                ns.append(n)
                regrets.append(r)
        
        if efficiencies:  # 确保有数据点
            color = colors[i]
            ax2.scatter(efficiencies, regrets, s=80, marker=markers[i % len(markers)],
                      color=color, alpha=0.7, label=name)
            
            # 添加N值标签
            for j, (eff, reg, n) in enumerate(zip(efficiencies, regrets, ns)):
                ax2.annotate(f'N={n}', (eff, reg), xytext=(5, 0), 
                           textcoords='offset points', fontsize=8)
            
            # 尝试绘制趋势线
            if len(efficiencies) >= 3:  # 至少需要3个点来拟合曲线
                try:
                    z = np.polyfit(efficiencies, regrets, 2)  # 二次多项式拟合
                    p = np.poly1d(z)
                    x_trend = np.linspace(min(efficiencies), max(efficiencies), 100)
                    ax2.plot(x_trend, p(x_trend), '--', color=color, alpha=0.5)
                except:
                    pass  # 如果拟合失败，就跳过
    
    # 设置第二个子图的属性
    ax2.set_xlabel('探索效率 (N/T)', fontsize=14)
    ax2.set_ylabel('平均后悔值', fontsize=14)
    ax2.set_title('探索效率与性能关系', fontsize=16)
    ax2.grid(True, linestyle='--', alpha=0.7)
    if len(strategy_data) <= 5:  # 只有当策略数量不多时才显示图例
        ax2.legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    filename = f'{filename_prefix}_combined.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  N值影响综合分析图已保存至: {filename}")
    
    # 额外创建一个简化版的图表，只显示T-N关系
    plt.figure(figsize=(10, 8))
    
    for i, (name, points) in enumerate(strategy_data.items()):
        points.sort(key=lambda x: x[0])  # 按N排序
        n_values = [p[0] for p in points]
        avg_t_values = [p[1] for p in points]
        
        plt.plot(n_values, avg_t_values, marker=markers[i % len(markers)], 
                 linestyle='-', color=colors[i], linewidth=2, label=name)
    
    plt.xscale('log')
    plt.xlabel('承诺阶段长度 (N)', fontsize=14)
    plt.ylabel('平均探索轮数 (T)', fontsize=14)
    plt.title('N值对动态策略探索轮数的影响', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best', fontsize=12)
    
    plt.tight_layout()
    filename = f'{filename_prefix}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  N值影响基础图已保存至: {filename}")
