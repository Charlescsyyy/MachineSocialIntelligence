# src/train.py
import torch
import torch.nn.functional as F
import numpy as np
from src.policy import PolicyNetwork
from src.env_interface import AdaptiveLanderEnv

# 超参数设置
learning_rate = 3e-4
gamma = 0.99
lam = 0.95
clip_eps = 0.2
value_coef = 1.0
entropy_coef = 0.01
train_epochs = 1000    # PPO更新迭代次数
batch_size = 2048      # 每次PPO更新收集的交互步数
minibatch_size = 256   # PPO每轮的小批量大小
ppo_epochs = 10        # 每次更新的训练epoch数

# 选择设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化环境和策略网络
env = AdaptiveLanderEnv()
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
policy_net = PolicyNetwork(obs_dim, act_dim).to(device)
optimizer = torch.optim.Adam(policy_net.parameters(), lr=learning_rate)

# 日志：保存每回合总奖励用于监控
episode_rewards = []
# 经验缓冲
obs_buffer, act_buffer = [], []
logp_buffer, val_buffer = [], []
rew_buffer, done_buffer = [], []

# 重置环境
obs = env.reset()
ep_reward = 0.0

# 主训练循环
for update in range(train_epochs):
    steps_collected = 0
    # 收集batch_size步的交互数据
    while steps_collected < batch_size:
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        action_tensor, log_prob_tensor, value_tensor = policy_net.act(obs_tensor)
        # 提取动作、log概率、状态价值
        action = action_tensor[0]                 # 动作 shape: (2,)
        log_prob = log_prob_tensor.item()
        value = value_tensor.item()
        # 环境一步交互
        next_obs, reward, done, info = env.step(action)
        # 存储经验
        obs_buffer.append(obs)
        act_buffer.append(action)
        logp_buffer.append(log_prob)
        val_buffer.append(value)
        rew_buffer.append(reward)
        done_buffer.append(done)
        ep_reward += reward
        steps_collected += 1
        obs = next_obs
        if done:
            episode_rewards.append(ep_reward)
            ep_reward = 0.0
            obs = env.reset()
    # 收集完一批数据后，计算优势和回报，并进行PPO更新
    obs_arr = np.array(obs_buffer, dtype=np.float32)
    act_arr = np.array(act_buffer, dtype=np.float32)
    old_logp_arr = np.array(logp_buffer, dtype=np.float32)
    val_arr = np.array(val_buffer, dtype=np.float32)
    rew_arr = np.array(rew_buffer, dtype=np.float32)
    done_arr = np.array(done_buffer, dtype=np.bool_)
    # 计算最后状态的价值（如果最后一个数据未终止，则用于bootstrap）
    last_val = 0.0
    if not done_arr[-1]:
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
        _, _, last_val_t = policy_net.forward(obs_tensor)
        last_val = last_val_t.item()
    # GAE优势估计
    advantages = np.zeros_like(rew_arr)
    returns = np.zeros_like(rew_arr)
    gae = 0.0
    for t in reversed(range(len(rew_arr))):
        if done_arr[t]:
            delta = rew_arr[t] - val_arr[t]
            gae = delta  # episode结束时，将GAE初始化为delta
        else:
            next_val = last_val if t == len(rew_arr) - 1 else (0.0 if done_arr[t+1] else val_arr[t+1])
            delta = rew_arr[t] + gamma * next_val - val_arr[t]
            gae = delta + gamma * lam * gae
        advantages[t] = gae
        returns[t] = advantages[t] + val_arr[t]
        # 如果此步是终止步，重置gae和last_val
        if done_arr[t]:
            gae = 0.0
            last_val = 0.0
        else:
            last_val = val_arr[t]
    # 优势标准化
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    # 转为tensor
    obs_tensor = torch.tensor(obs_arr, dtype=torch.float32).to(device)
    act_tensor = torch.tensor(act_arr, dtype=torch.float32).to(device)
    old_logp_tensor = torch.tensor(old_logp_arr, dtype=torch.float32).to(device)
    adv_tensor = torch.tensor(advantages, dtype=torch.float32).to(device)
    ret_tensor = torch.tensor(returns, dtype=torch.float32).to(device)
    # PPO策略与价值更新
    num_samples = len(obs_arr)
    indices = np.arange(num_samples)
    for epoch in range(ppo_epochs):
        np.random.shuffle(indices)
        for start in range(0, num_samples, minibatch_size):
            end = start + minibatch_size
            mb_idx = indices[start:end]
            mb_states = obs_tensor[mb_idx]
            mb_actions = act_tensor[mb_idx]
            mb_old_logp = old_logp_tensor[mb_idx]
            mb_adv = adv_tensor[mb_idx]
            mb_returns = ret_tensor[mb_idx]
            new_logp, entropy, new_value = policy_net.evaluate_actions(mb_states, mb_actions)
            new_value = new_value.squeeze(-1)
            # 计算PPO目标：CLIP策略损失 + 价值函数损失 + 熵正则
            ratio = torch.exp(new_logp - mb_old_logp)
            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * mb_adv
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(new_value, mb_returns)
            entropy_loss = -entropy.mean()
            loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    # 清空经验缓冲区，为下一个batch做准备
    obs_buffer.clear()
    act_buffer.clear()
    logp_buffer.clear()
    val_buffer.clear()
    rew_buffer.clear()
    done_buffer.clear()
    # 打印训练进度
    if (update + 1) % 10 == 0:
        avg_reward = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else np.mean(episode_rewards)
        print(f"更新 {update+1}/{train_epochs}，最近10回合平均回报: {avg_reward:.2f}")
# 保存训练后的模型参数
torch.save(policy_net.state_dict(), "ppo_assistant_model.pth")
print("训练完成，模型已保存为 ppo_assistant_model.pth")