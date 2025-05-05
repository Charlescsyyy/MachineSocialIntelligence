# src/env_interface.py
import gym
import numpy as np
from src.planning import ParticleFilter
from src.reward import empowerment_reward, task_efficiency_reward

class AdaptiveLanderEnv(gym.Env):
    '''
    自定义环境：包装LunarLanderContinuous，以模拟人类-助理共享控制：
    包含人类飞行员策略、助理策略作用、意图推断和自适应奖励计算。
    '''
    def __init__(self, possible_goals=[-0.5, 0.0, 0.5], max_steps=1000):
        # 基础环境初始化
        self.env = gym.make("LunarLanderContinuous-v2")
        self.possible_goals = np.array(possible_goals, dtype=float)
        self.max_steps = max_steps
        # 粒子滤波器和目标初始化
        self.particle_filter = None
        self.goal = None
        # 用于保存上一次状态和用户动作
        self.last_state = None
        self.last_user_action = None
        # 定义动作空间和观测空间
        self.action_space = self.env.action_space  # 助理动作空间与原环境相同
        obs_dim = self.env.observation_space.shape[0]  # 原始状态维度 (8)
        obs_dim_extended = obs_dim + 2 + len(self.possible_goals)  # 扩展后的观测维度
        high = np.inf * np.ones(obs_dim_extended, dtype=np.float32)
        low = -high
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        # 计步器
        self.step_count = 0
    
    def reset(self):
        # 环境重置，随机生成本episode的真实目标
        base_state = self.env.reset()
        self.goal = float(np.random.choice(self.possible_goals))
        # 重置粒子滤波器（意图分布初始均匀）
        self.particle_filter = ParticleFilter(self.possible_goals, num_particles=100)
        self.step_count = 0
        # 计算初始用户动作（假设用户在初始状态看到目标，立即采取行动）
        user_action = self._pilot_policy(base_state, self.goal)
        self.last_state = base_state
        self.last_user_action = user_action
        # 根据初始用户动作更新意图分布
        self.particle_filter.update(base_state, user_action)
        belief = self.particle_filter.get_distribution()
        # 构建返回的初始观测（状态 + 用户动作 + 目标分布）
        obs = np.concatenate([base_state, user_action, belief]).astype(np.float32)
        return obs
    
    def step(self, assist_action):
        # 获取当前状态和用户动作
        state = self.last_state
        user_action = self.last_user_action
        # 助理动作（numpy数组）
        assist_action = np.array(assist_action, dtype=np.float32)
        # 融合用户和助理动作（简单相加并截断到[-1,1]范围）
        final_action = np.clip(user_action + assist_action, -1.0, 1.0)
        # 应用组合动作推进环境一个时间步
        new_state, env_reward, done, info_env = self.env.step(final_action)
        self.step_count += 1
        # 若达到最大步数且未结束，则强制终止（超时）
        if self.step_count >= self.max_steps:
            done = True
        # 计算下一时刻用户的动作（基于新的状态和既定目标）
        next_user_action = self._pilot_policy(new_state, self.goal)
        # 用新状态下用户动作更新粒子滤波器（更新意图分布）
        self.particle_filter.update(new_state, next_user_action)
        belief = self.particle_filter.get_distribution()
        # 计算奖励：赋能 vs 任务效率 动态加权
        entropy = -np.sum(belief * np.log(belief + 1e-8))
        max_entropy = np.log(len(belief))
        weight = entropy / max_entropy if max_entropy > 0 else 0.0  # 当前不确定性的归一化程度
        emp_r = empowerment_reward(state, self.possible_goals)
        task_r = task_efficiency_reward(new_state, self.goal, done)
        reward = weight * emp_r + (1 - weight) * task_r
        # 更新最近状态和用户动作
        self.last_state = new_state
        self.last_user_action = next_user_action
        # 构建新的观测
        obs = np.concatenate([new_state, next_user_action, belief]).astype(np.float32)
        # info中提供真实目标和当前意图分布，便于评估和调试
        info = {'goal': self.goal, 'belief': belief}
        if done:
            # 标记成功与否（根据最终任务奖励或状态判断）
            success = False
            if task_r > 0:  # 如果最后一步任务奖励为正，表示成功着陆在目标
                success = True
            info['success'] = success
        return obs, reward, done, info
    
    def _pilot_policy(self, state, goal_x):
        '''
        模拟人类飞行员的控制策略：根据目标位置对飞船施加近似控制。
        返回值为连续动作(np.array([主引擎推力, 横向推力]))。
        '''
        x = state[0]; y = state[1]
        vx = state[2]; vy = state[3]
        main_thruster = 0.0
        lateral_thruster = 0.0
        # 水平控制：根据目标相对位置决定左右侧推
        if goal_x - x > 0.1:
            # 目标在右侧，需要向右移动 -> 点火左侧推进行星（横向动作为负）
            lateral_thruster = -1.0
        elif goal_x - x < -0.1:
            # 目标在左侧，需要向左移动 -> 点火右侧推进行星（横向动作为正）
            lateral_thruster = 1.0
        else:
            lateral_thruster = 0.0
        # 如果水平速度方向与目标方向相反，则施加反向侧推以减速
        if (goal_x - x) * vx < 0:
            lateral_thruster = -np.sign(vx)
        # 垂直控制：控制下降速度，接近地面时逐渐减速
        if y > 0.5:
            # 高空时允许下落，若下落过快则稍微推力减速
            if vy < -1.0:
                main_thruster = 0.5
        else:
            # 低空时确保软着陆：下降过快则强推，轻微下降则小推，缓降或上升则不推
            if vy < -0.5:
                main_thruster = 1.0
            elif vy < -0.1:
                main_thruster = 0.5
            else:
                main_thruster = 0.0
        return np.array([main_thruster, lateral_thruster], dtype=np.float32)