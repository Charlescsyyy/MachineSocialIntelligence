# src/planning.py
import numpy as np
import math

class ParticleFilter:
    def __init__(self, possible_goals, num_particles=100, neural_model=None):
        '''
        粒子滤波器初始化。
        possible_goals: 可能的目标列表（离散值）或连续范围（数组长度为2表示[min, max]）
        num_particles: 粒子数量
        neural_model: 可选的神经网络模型用于提供目标先验概率（实现神经引导）
        '''
        self.possible_goals = np.array(possible_goals)
        self.num_particles = num_particles
        self.neural_model = neural_model
        # 初始化粒子集合：根据目标空间离散集合均匀抽样，或在连续区间均匀采样
        if self.possible_goals.ndim == 1:
            # 离散目标值集合
            self.particles = np.random.choice(self.possible_goals, size=num_particles)
        else:
            # 连续目标范围 (假定possible_goals = [min, max])
            low, high = self.possible_goals[0], self.possible_goals[1]
            self.particles = np.random.uniform(low, high, size=num_particles)
        # 初始权重均匀分布
        self.weights = np.ones(num_particles) / num_particles
    
    def update(self, state, user_action):
        '''
        基于当前观测(状态和用户动作)更新粒子权重并重采样，推断用户目标分布
        state: 当前环境状态（numpy数组）
        user_action: 用户在该状态下执行的动作（numpy数组）
        '''
        x = state[0]
        vx = state[2]
        lateral_act = user_action[1]
        new_weights = np.zeros(self.num_particles)
        sigma = 0.5  # 人类动作噪声标准差（假设）
        # 计算每个粒子对应目标下，该用户动作的似然
        for i, goal in enumerate(self.particles):
            # 理想的用户横向动作（基于简单的目标定向策略）
            target_offset_x = goal - x
            ideal_lateral = 0.0
            if target_offset_x > 0.1:
                ideal_lateral = -1.0   # 目标在右侧 -> 理想操作：左推（负侧推）
            elif target_offset_x < -0.1:
                ideal_lateral = 1.0    # 目标在左侧 -> 理想操作：右推（正侧推）
            else:
                ideal_lateral = 0.0    # 已接近目标水平位置
            # 若当前横向速度朝远离目标方向，则理想动作应朝反方向减速
            if target_offset_x * vx < 0:
                ideal_lateral = -np.sign(vx)
            # 用户实际横向动作与理想动作的差异
            diff = lateral_act - ideal_lateral
            # 根据高斯模型计算该差异的似然（差异越小，似然越高）
            likelihood = math.exp(-(diff**2) / (2 * sigma**2))
            new_weights[i] = likelihood * self.weights[i]
        # 融合神经网络先验（若提供）：利用神经模型预测的目标概率调整权重
        if self.neural_model is not None:
            # 假设神经网络模型提供 possible_goals 每个目标的概率列表
            nn_probs = self.neural_model.predict(state)
            for i, goal in enumerate(self.particles):
                if goal in self.possible_goals:
                    idx = np.where(self.possible_goals == goal)[0][0]
                    new_weights[i] *= nn_probs[idx]
        # 归一化权重
        total = new_weights.sum()
        if total == 0:
            # 若所有粒子的似然为0（极端情况），重置为均匀分布避免数值问题
            new_weights = np.ones(self.num_particles) / self.num_particles
        else:
            new_weights /= total
        self.weights = new_weights
        # 粒子重采样：根据更新后的权重分布选择新的粒子集合
        indices = np.random.choice(self.num_particles, size=self.num_particles, p=self.weights)
        self.particles = self.particles[indices]
        # 重采样后重置权重为均匀（每个粒子权重相等）
        self.weights.fill(1.0 / self.num_particles)
    
    def get_distribution(self):
        '''
        返回根据当前粒子集合估计的目标概率分布（与possible_goals顺序对应）
        '''
        dist = np.zeros(len(self.possible_goals))
        for p in self.particles:
            # 统计粒子对应各目标值的频率
            idx = np.where(self.possible_goals == p)[0]
            if idx.size > 0:
                dist[idx[0]] += 1
        dist = dist / self.num_particles
        return dist