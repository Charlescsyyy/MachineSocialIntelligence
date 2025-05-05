#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GoalInferenceEnv
----------------
把底层 LunarLanderEmpowerment 环境封装成 “猜目标编号” 的 Gym 环境，
以便用 PPO2／DQN 等离散控制算法训练目标推断策略。

观测 (obs): belief(k) + lander_state(8) + human_action(2)  =>  k + 10 维
动作 (act): 预测目标编号 0..k-1                      (Discrete k)
奖励 (rew): -KL(b_t‖b_{t+1})       +  终局猜中(+1) / 猜错(+0)
"""

import gym
import numpy as np


class GoalInferenceEnv(gym.Env):
    def __init__(self, base_env, k_goals: int = 2):
        """
        Parameters
        ----------
        base_env : LunarLanderEmpowerment
            已经初始化好的底层环境实例
        k_goals : int, default=2
            可能目标数量；应与 base_env.k_goals 保持一致
        """
        super().__init__()
        self.base = base_env
        self.k_goals = k_goals

        self.action_space = gym.spaces.Discrete(k_goals)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(k_goals + 10,),
            dtype=np.float32
        )
        # belief 初始均匀
        self.belief = np.ones(k_goals, dtype=np.float32) / k_goals

    # --------------------------------------------------
    # Gym API
    # --------------------------------------------------
    def reset(self, **kw):
        """重置底层环境并返回打包后的观测"""
        self.belief[:] = 1.0 / self.k_goals
        obs = self.base.reset(**kw)
        return self._pack(obs)

    def step(self, action: int):
        """
        Parameters
        ----------
        action : int
            当前帧对目标的预测编号 0..k-1
        Returns
        -------
        obs  : np.ndarray, shape=(k_goals+10,)
        rew  : float
        done : bool
        info : dict
        """
        # ---------- 让底层环境跑一步（不控制飞船） ----------
        if getattr(self.base.action_space, 'shape', None):     # 连续
            idle_action = np.zeros(self.base.action_space.shape,
                                   dtype=self.base.action_space.dtype)
        else:                                                  # 离散
            idle_action = 1       # encode_human_action([0,1]) -> no thruster
        obs, _, done, info = self.base.step(idle_action)

        # ---------- belief 更新与即时奖励 ----------
        new_belief = info.get("belief", self.belief)
        kl = float((self.belief *
                    (np.log(self.belief + 1e-8) -
                     np.log(new_belief + 1e-8))).sum())
        reward = -kl
        self.belief = new_belief

        # ---------- 终局奖励 ----------
        if done and info.get("true_goal") is not None:
            reward += 1.0 if action == int(info["true_goal"]) else 0.0

        return self._pack(obs), reward, done, info

    # --------------------------------------------------
    # 打包观测
    # --------------------------------------------------
    def _pack(self, obs):
        human_act = getattr(self.base, "last_human_action", (0.0, 1.0))
        return np.concatenate([self.belief, obs[:8], human_act]).astype(np.float32)

