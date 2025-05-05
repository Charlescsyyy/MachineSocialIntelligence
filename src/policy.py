# src/policy.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        # Actor网络（输出动作均值）
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh()  # 输出使用tanh将动作均值限制在[-1,1]范围
        )
        # Critic网络（输出状态价值）
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        # 可学习的对数标准差参数（用于高斯策略噪声）
        self.log_std = nn.Parameter(torch.zeros(action_dim))
    
    def forward(self, state):
        '''返回actor输出的动作均值和critic输出的状态价值'''
        mean = self.actor(state)
        value = self.critic(state)
        # 将log_std转换为标准差（确保正值）
        std = torch.exp(self.log_std)
        return mean, std, value
    
    def act(self, state):
        '''根据当前策略对给定状态采样动作，并返回动作、log概率和状态价值（用于交互）'''
        mean, std, value = self.forward(state)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()                     # 从高斯分布采样动作
        log_prob = dist.log_prob(action).sum(dim=-1)  # 计算动作的对数概率（多维求和）
        # 返回numpy格式的动作，以及tensor格式的log概率和值
        return action.detach().cpu().numpy(), log_prob.detach(), value.detach()
    
    def evaluate_actions(self, state, action):
        '''计算给定状态和动作下策略的log概率、熵和状态价值（用于PPO训练时的评价）'''
        mean, std, value = self.forward(state)
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1)   # 动作log概率
        entropy = dist.entropy().sum(dim=-1)           # 策略熵
        return log_prob, entropy, value