# src/reward.py
import numpy as np

def empowerment_reward(state, possible_goals):
    '''
    计算赋能式奖励：鼓励在目标不明确时保持状态的可控性和多样性，
    例如位于各潜在目标之间较高的位置，提高后续到达任意目标的可能性。
    '''
    x = state[0]
    y = state[1]
    # 计算与每个潜在目标的水平距离
    distances = [abs(g - x) for g in possible_goals]
    if len(distances) == 0:
        return 0.0
    spread = max(distances) - min(distances)  # 距离差，用于衡量偏向某目标的程度
    # 奖励更高的高度（y）和更小的距离差（表示位于各目标中间）
    emp_reward = 0.1 * y - spread
    return emp_reward

def task_efficiency_reward(state, goal_x, done):
    '''
    计算任务效率奖励：鼓励快速准确地达到用户目标。
    包括时间惩罚、距离惩罚，以及成功/失败的大额奖惩。
    '''
    x = state[0]; y = state[1]
    vx = state[2]; vy = state[3]
    reward = 0.0
    if not done:
        # 尚未结束时，每步惩罚时间和水平误差，以促使尽快接近目标
        reward -= 1.0                  # 时间惩罚（每步 -1）
        reward -= 0.5 * abs(goal_x - x)  # 水平距离惩罚（距离目标越远，惩罚越大）
    else:
        # Episode结束时，根据着陆结果给予奖励或惩罚
        leg1_contact = state[6]; leg2_contact = state[7]
        tolerance = 0.1  # 着陆水平容差
        # 判断是否双腿着陆且在目标位置附近
        if leg1_contact > 0.5 and leg2_contact > 0.5 and abs(goal_x - x) < tolerance:
            reward += 100.0  # 成功安全地在目标附近着陆
        else:
            reward -= 100.0  # 未能在目标处安全着陆（包括坠毁或偏离目标）
    return reward