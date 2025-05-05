# src/evaluate.py
import torch
import numpy as np
from src.policy import PolicyNetwork
from src.env_interface import AdaptiveLanderEnv
from src.reward import empowerment_reward

# 加载训练好的模型参数
env = AdaptiveLanderEnv()
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
policy_net = PolicyNetwork(obs_dim, act_dim)
policy_net.load_state_dict(torch.load("ppo_assistant_model.pth", map_location=torch.device('cpu')))
policy_net.eval()

# 评估设置
num_episodes = 50
success_times = []   # 成功着陆所用步数列表
error_count = 0      # 错误辅助出现的episode计数
emp_scores = []      # 每回合赋能评分
timing_scores = []   # 每回合自适应时机评分
possible_goals = env.possible_goals

for episode in range(num_episodes):
    obs = env.reset()
    done = False
    step_count = 0
    episode_emp_rewards = []
    assist_actions = []
    belief_history = []
    true_goal = None
    # 执行一个episode
    while not done:
        state = obs[:env.env.observation_space.shape[0]]  # 提取基础状态部分
        # 使用确定性策略（actor均值）选择动作
        state_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            mean, std, value = policy_net.forward(state_tensor)
        action = mean.squeeze(0).numpy()  # 取均值动作
        # 计算当前步赋能奖励（用于统计赋能分数）
        emp_r = empowerment_reward(state, possible_goals)
        episode_emp_rewards.append(emp_r)
        # 应用动作与环境交互
        obs, reward, done, info = env.step(action)
        assist_actions.append(action)
        belief = info.get('belief')
        if belief is not None:
            belief_history.append(belief)
        true_goal = info.get('goal', true_goal)
        step_count += 1
    # 判断成功与否并记录完成时间
    success = info.get('success', False)
    if success:
        success_times.append(step_count)
    # 判断错误辅助：检查助理是否在过程中朝相反方向用力（暗示早期目标判断错误）
    assist_actions = np.array(assist_actions)
    lateral_actions = assist_actions[:, 1]  # 取横向动作分量
    if lateral_actions.size > 0:
        if lateral_actions.max() > 0.5 and lateral_actions.min() < -0.5:
            error_count += 1  # 存在明显的左右方向大幅度切换
        elif not success:
            # 若未成功，则视为一次错误辅助
            error_count += 1
    # 计算该回合赋能分数（赋能奖励的平均值，可衡量赋能支持程度）
    emp_scores.append(np.mean(episode_emp_rewards))
    # 计算该回合自适应时机评分
    timing_score = 0.0
    if success and belief_history:
        # 找到实际目标在belief分布中的索引
        if true_goal is not None:
            try:
                goal_idx = list(possible_goals).index(true_goal)
            except ValueError:
                # 如目标不在预定义列表（连续情况），取最后一步分布最大概率索引代替
                goal_idx = int(np.argmax(belief_history[-1]))
        else:
            goal_idx = int(np.argmax(belief_history[-1]))
        # 找到首次对真实目标概率超过阈值的时间步
        switch_step = len(belief_history)  # 默认为最后
        threshold = 0.8
        for t, belief in enumerate(belief_history):
            if belief[goal_idx] > threshold:
                switch_step = t
                break
        # 根据切换时刻占整个任务的比例计算评分：越早确定目标，评分越高
        timing_score = max(0.0, 1.0 - switch_step / len(belief_history))
    else:
        timing_score = 0.0
    timing_scores.append(timing_score)

# 计算平均指标
avg_time = np.mean(success_times) if success_times else float('inf')
error_rate = error_count / num_episodes
avg_emp_score = np.mean(emp_scores)
avg_timing_score = np.mean(timing_scores)
# 输出评估结果
print(f"平均任务完成时间: {avg_time:.2f} 步")
print(f"错误辅助率: {error_rate:.2f}")
print(f"平均赋能分数: {avg_emp_score:.2f}")
print(f"平均自适应时机评分: {avg_timing_score:.2f}")