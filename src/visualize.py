# src/visualize.py
import numpy as np
import matplotlib.pyplot as plt

def plot_strategy_switch(belief_history, possible_goals, true_goal):
    '''
    绘制策略切换过程：展示每个潜在目标的概率随时间的变化，并标注切换点和阈值线。
    '''
    belief_history = np.array(belief_history)
    T, G = belief_history.shape
    for i in range(G):
        plt.plot(belief_history[:, i], label=f"Goal {possible_goals[i]}")
    # 绘制判定切换的概率阈值线（例如0.8）
    plt.axhline(0.8, color='gray', linestyle='--', label='Confidence 0.8')
    # 标注实际目标的概率曲线为突出线
    if true_goal in possible_goals:
        true_idx = list(possible_goals).index(true_goal)
        plt.plot(belief_history[:, true_idx], color='red', linewidth=2, label=f"True Goal {true_goal}")
    # 查找助理策略切换时刻（实际目标概率首次超过阈值）
    switch_step = None
    if true_goal in possible_goals:
        true_idx = list(possible_goals).index(true_goal)
        for t in range(T):
            if belief_history[t, true_idx] > 0.8:
                switch_step = t
                break
    if switch_step is not None:
        plt.axvline(switch_step, color='red', linestyle='--', label='Switch Point')
    plt.xlabel("Time Step")
    plt.ylabel("Probability")
    plt.title("Goal Belief Over Time")
    plt.legend()
    plt.show()

def plot_performance_metrics(avg_time, error_rate, avg_empowerment, avg_timing):
    '''
    绘制性能指标柱状图：包括平均任务完成时间、错误辅助率、赋能分数、自适应时机评分。
    '''
    metrics = {
        "Avg Time": avg_time,
        "Error Rate": error_rate,
        "Empowerment": avg_empowerment,
        "Timing": avg_timing
    }
    names = list(metrics.keys())
    values = list(metrics.values())
    plt.figure(figsize=(6,4))
    plt.bar(names, values, color=['blue','orange','green','purple'])
    plt.title("Performance Metrics")
    for i, v in enumerate(values):
        plt.text(i, v + 0.01, f"{v:.2f}", ha='center', va='bottom')
    plt.ylim([0, max(values)*1.2])
    plt.show()