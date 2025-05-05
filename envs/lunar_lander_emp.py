#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Lunar Lander with Empowerment + (optional) Bayesian Goal Belief
----------------------------------------------------------------
* empowerment       —— 信息论稳定奖励（与原仓库一致）
* Bayesian belief   —— 2-goal online update, exposed via info['belief']
* goal_predictor    —— 若上层注入，可用于 autopilot 切换等
"""


from __future__ import division
import sys, math, os, csv
from copy import deepcopy

import numpy as np
import Box2D
from Box2D.b2 import (edgeShape, circleShape, fixtureDef,
                       polygonShape, revoluteJointDef, contactListener)
import gym
from typing import Optional
from gym import spaces
from gym.utils import EzPickle, seeding

from utils.env_utils import disc_to_cont, onehot_encode

# ===== 常量 =====
MAX_NUM_STEPS = 1000
OBS_DIM = 9
ACT_DIM = 6

FPS = 50
SCALE = 30.0
MAIN_ENGINE_POWER = 13.0
SIDE_ENGINE_POWER = 0.6
INITIAL_RANDOM = 1000.0

LANDER_POLY = [
    (-14, +17), (-17, 0), (-17, -10),
    (+17, -10), (+17, 0), (+14, +17)
]
LEG_AWAY, LEG_DOWN = 20, 18
LEG_W, LEG_H = 2, 8
LEG_SPRING_TORQUE = 40
SIDE_ENGINE_HEIGHT, SIDE_ENGINE_AWAY = 14.0, 12.0

VIEWPORT_W, VIEWPORT_H = 600, 400
NUM_CONCAT = 20
# =====================================================


class ContactDetector(contactListener):
    """检测机身碰撞 / 腿部接地"""
    def __init__(self, env):
        super().__init__()
        self.env = env

    def BeginContact(self, contact):
        if self.env.lander in [contact.fixtureA.body, contact.fixtureB.body]:
            self.env.game_over = True
        for i in range(2):
            if self.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                self.env.legs[i].ground_contact = True

    def EndContact(self, contact):
        for i in range(2):
            if self.env.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                self.env.legs[i].ground_contact = False


class LunarLanderEmpowerment(gym.Env, EzPickle):
    """双跑道 + Empowerment + （可选）Bayesian belief"""
    metadata = {'render.modes': ['human', 'rgb_array'],
                'video.frames_per_second': FPS}

    continuous = False  # 默认离散动作

    # ------------------------------------------------------------------
    # 初始化
    # ------------------------------------------------------------------
    def __init__(self,
                 empowerment: float = 0.0,
                 ac_continuous: bool = False,
                 pilot_policy=None,
                 pilot_is_human: bool = False,
                 log_file: Optional[str] = None,
                 goal_predictor=None,
                 k_goals: int = 2,
                 **extras):
        EzPickle.__init__(self, empowerment, ac_continuous)
        self.seed()
        self.viewer = None

        # ---------- belief / goal ----------
        self.k_goals = k_goals
        self.belief = np.ones(k_goals, dtype=np.float32) / k_goals
        self.true_goal = np.random.randint(k_goals)
        self.goal_predictor = goal_predictor

        # ---------- 其他成员 ----------
        self.num_concat = NUM_CONCAT
        self.act_dim = ACT_DIM

        self.world = Box2D.b2World()
        self.moon = None
        self.lander = None
        self.legs: list = []
        self.particles: list = []

        self.pilot_is_human = pilot_is_human
        self.copilot = pilot_policy is not None
        self.pilot_policy = pilot_policy
        if self.copilot:
            self.past_pilot_actions = np.zeros(NUM_CONCAT * ACT_DIM)

        self.continuous = ac_continuous
        self.empowerment_coeff = empowerment
        self.fake_step = False
        self.curr_step = 0

        self.log_file = log_file or os.path.join(os.getcwd(), 'empowerment_log')

        # 观测空间
        obs_box = spaces.Box(-np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float32)
        if self.copilot:
            self.observation_space = spaces.Box(
                np.concatenate((obs_box.low, np.zeros(NUM_CONCAT * ACT_DIM))),
                np.concatenate((obs_box.high, np.ones(NUM_CONCAT * ACT_DIM))))
        else:
            self.observation_space = obs_box

        # 动作空间
        self.action_space = (spaces.Box(-1.0, +1.0, (2,), dtype=np.float32)
                             if self.continuous
                             else spaces.Discrete(ACT_DIM))

        # 初始化世界
        self.reset()

    # ------------------------------------------------------------------
    # 贝叶斯 belief 更新（简单似然：看侧推键）
    # ------------------------------------------------------------------
    def _update_belief(self, human_action):
        """
        human_action[1]==0 → 左键 → 偏向 goal 0
        human_action[1]==2 → 右键 → 偏向 goal 1
        neutral           → 不改变
        """
        if self.k_goals != 2:
            return
        lat = human_action[1]
        if lat == 0:
            likelihood = np.array([0.8, 0.2], dtype=np.float32)
        elif lat == 2:
            likelihood = np.array([0.2, 0.8], dtype=np.float32)
        else:
            likelihood = np.ones(2, dtype=np.float32)
        self.belief *= likelihood
        self.belief /= self.belief.sum()

    # ------------------------------------------------------------------
    # 销毁世界
    # ------------------------------------------------------------------
    def _destroy(self):
        if not self.moon:
            return
        self.world.contactListener = None
        self._clean_particles(True)
        self.world.DestroyBody(self.moon)
        self.world.DestroyBody(self.lander)
        self.world.DestroyBody(self.legs[0])
        self.world.DestroyBody(self.legs[1])
        self.moon = self.lander = None
        self.legs = []

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(self):
        self._destroy()

        # belief 重置
        self.belief[:] = 1.0 / self.k_goals
        self.true_goal = self.np_random.randint(self.k_goals)

        # 监听器
        self.world.contactListener_keepref = ContactDetector(self)
        self.world.contactListener = self.world.contactListener_keepref

        # 状态量
        self.game_over = False
        self.prev_shaping = None
        self.curr_step = 0
        self.num_steps_at_site = 0
        self.trajectory, self.actions = [], []

        # 世界尺寸
        W, H = VIEWPORT_W / SCALE, VIEWPORT_H / SCALE
        CHUNKS = 11

        # ---------- 地形 ----------
        height = self.np_random.uniform(0, H / 2, size=(CHUNKS + 1,))
        chunk_x = [W / (CHUNKS - 1) * i for i in range(CHUNKS)]

        pad_y = H / 4
        c1 = self.np_random.choice(range(2, CHUNKS - 2))
        c2_candidates = [c for c in range(2, CHUNKS - 2) if abs(c - c1) > 2]
        c2 = self.np_random.choice(c2_candidates)

        self.helipads = [
            (chunk_x[c1 - 1], chunk_x[c1 + 1]),
            (chunk_x[c2 - 1], chunk_x[c2 + 1])
        ]
        (self.helipad_x1, self.helipad_x2) = self.helipads[0]
        (self.helipad2_x1, self.helipad2_x2) = self.helipads[1]
        self.helipad_y = pad_y

        for c in (c1, c2):
            for k in range(c - 2, c + 3):
                height[k] = pad_y

        smooth_y = [0.33 * (height[i - 1] + height[i] + height[i + 1])
                    for i in range(CHUNKS)]

        # --------- 地面静态体 ---------
        self.moon = self.world.CreateStaticBody(
            shapes=edgeShape(vertices=[(0, 0), (W, 0)]))
        self.sky_polys: list = []
        for i in range(CHUNKS - 1):
            p1, p2 = (chunk_x[i], smooth_y[i]), (chunk_x[i + 1], smooth_y[i + 1])
            self.moon.CreateEdgeFixture(vertices=[p1, p2], density=0, friction=0.1)
            self.sky_polys.append([p1, p2, (p2[0], H), (p1[0], H)])
        self.moon.color1 = self.moon.color2 = (0, 0, 0)

        # --------- 飞船 ---------
        initial_y = VIEWPORT_H / SCALE
        self.lander = self.world.CreateDynamicBody(
            position=(VIEWPORT_W / SCALE / 2, initial_y),
            angle=0.0,
            fixtures=fixtureDef(
                shape=polygonShape(vertices=[(x / SCALE, y / SCALE)
                                             for x, y in LANDER_POLY]),
                density=5.0,
                friction=0.1,
                categoryBits=0x0010,
                maskBits=0x001,
                restitution=0.0))
        self.lander.color1, self.lander.color2 = (0.5, 0.4, 0.9), (0.3, 0.3, 0.5)
        self.lander.ApplyForceToCenter(
            (self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM),
             self.np_random.uniform(-INITIAL_RANDOM, INITIAL_RANDOM)), True)

        # --------- 腿 ---------
        self.legs = []
        for i in [-1, +1]:
            leg = self.world.CreateDynamicBody(
                position=(VIEWPORT_W / SCALE / 2 - i * LEG_AWAY / SCALE, initial_y),
                angle=0.05 * i,
                fixtures=fixtureDef(
                    shape=polygonShape(box=(LEG_W / SCALE, LEG_H / SCALE)),
                    density=1.0,
                    restitution=0.0,
                    categoryBits=0x0020,
                    maskBits=0x001))
            leg.color1, leg.color2 = (0.5, 0.4, 0.9), (0.3, 0.3, 0.5)
            rjd = revoluteJointDef(
                bodyA=self.lander, bodyB=leg,
                localAnchorA=(0, 0),
                localAnchorB=(i * LEG_AWAY / SCALE, LEG_DOWN / SCALE),
                enableMotor=True, enableLimit=True,
                maxMotorTorque=LEG_SPRING_TORQUE,
                motorSpeed=0.3 * i)
            rjd.lowerAngle, rjd.upperAngle = (+0.4, +0.9) if i == -1 else (-0.9, -0.4)
            leg.joint = self.world.CreateJoint(rjd)
            leg.ground_contact = False
            self.legs.append(leg)

        self.drawlist = [self.lander] + self.legs

        # dummy step 获取初始观测
        dummy = np.array([0.0, 0.0]) if self.continuous else 1
        return self.step(dummy)[0]

    # ------------------------------------------------------------------
    # 工具
    # ------------------------------------------------------------------
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _create_particle(self, mass, x, y, ttl):
        p = self.world.CreateDynamicBody(
            position=(x, y), angle=0.0,
            fixtures=fixtureDef(
                shape=circleShape(radius=2 / SCALE),
                density=mass, friction=0.1,
                categoryBits=0x0100, maskBits=0x001,
                restitution=0.3))
        p.ttl = ttl
        self.particles.append(p)
        self._clean_particles(False)
        return p

    def _clean_particles(self, all_particles):
        while self.particles and (all_particles or self.particles[0].ttl < 0):
            self.world.DestroyBody(self.particles.pop(0))

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def step(self, action):
        assert self.action_space.contains(action), f"{action} invalid"
        action = disc_to_cont(action)

        # ======== 发动机推力 ========
        tip = (math.sin(self.lander.angle), math.cos(self.lander.angle))
        side = (-tip[1], tip[0])
        dispersion = [self.np_random.uniform(-1.0, +1.0) / SCALE for _ in range(2)]

        m_power = s_power = 0.0

        # 主引擎
        if action[0] > 0.0:
            m_power = (np.clip(action[0], 0.0, 1.0) + 1.0) * 0.5   # 0.5..1
            ox = tip[0] * (4 / SCALE + 2 * dispersion[0]) + side[0] * dispersion[1]
            oy = -tip[1] * (4 / SCALE + 2 * dispersion[0]) - side[1] * dispersion[1]
            impulse_pos = (self.lander.position[0] + ox, self.lander.position[1] + oy)
            if not self.fake_step:
                p = self._create_particle(3.5, *impulse_pos, m_power)
                p.ApplyLinearImpulse((ox * MAIN_ENGINE_POWER * m_power,
                                      oy * MAIN_ENGINE_POWER * m_power),
                                     impulse_pos, True)
            self.lander.ApplyLinearImpulse((-ox * MAIN_ENGINE_POWER * m_power,
                                            -oy * MAIN_ENGINE_POWER * m_power),
                                           impulse_pos, True)

        # 侧向引擎
        if abs(action[1]) > 0.5:
            direction = np.sign(action[1])
            s_power = np.clip(abs(action[1]), 0.5, 1.0)
            ox = tip[0] * dispersion[0] + side[0] * (3 * dispersion[1] +
                                                     direction * SIDE_ENGINE_AWAY / SCALE)
            oy = -tip[1] * dispersion[0] - side[1] * (3 * dispersion[1] +
                                                      direction * SIDE_ENGINE_AWAY / SCALE)
            impulse_pos = (self.lander.position[0] + ox - tip[0] * 17 / SCALE,
                           self.lander.position[1] + oy + tip[1] * SIDE_ENGINE_HEIGHT / SCALE)
            if not self.fake_step:
                p = self._create_particle(0.7, *impulse_pos, s_power)
                p.ApplyLinearImpulse((ox * SIDE_ENGINE_POWER * s_power,
                                      oy * SIDE_ENGINE_POWER * s_power),
                                     impulse_pos, True)
            self.lander.ApplyLinearImpulse((-ox * SIDE_ENGINE_POWER * s_power,
                                            -oy * SIDE_ENGINE_POWER * s_power),
                                           impulse_pos, True)

        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)

        # ======== 状态向量 ========
        pos, vel = self.lander.position, self.lander.linearVelocity
        centers = [(x1 + x2) / 2 for (x1, x2) in self.helipads]
        helipad_x = min(centers, key=lambda c: abs(pos.x - c))

        state = [
            (pos.x - VIEWPORT_W / SCALE / 2) / (VIEWPORT_W / SCALE / 2),
            (pos.y - (self.helipad_y + LEG_DOWN / SCALE)) / (VIEWPORT_H / SCALE / 2),
            vel.x * (VIEWPORT_W / SCALE / 2) / FPS,
            vel.y * (VIEWPORT_H / SCALE / 2) / FPS,
            self.lander.angle,
            20.0 * self.lander.angularVelocity / FPS,
            1.0 if self.legs[0].ground_contact else 0.0,
            1.0 if self.legs[1].ground_contact else 0.0,
            (helipad_x - VIEWPORT_W / SCALE / 2) / (VIEWPORT_W / SCALE / 2)
        ]
        state = np.asarray(state, dtype=np.float32)

        # 记录
        self.curr_step += 1
        if not self.fake_step:
            self.trajectory.append(state)
            self.actions.append(action)

        # ======== Reward ========
        dx = (pos.x - helipad_x) / (VIEWPORT_W / SCALE / 2)
        shaping = -100 * np.hypot(state[2], state[3]) - 100 * abs(state[4]) \
                  + 10 * state[6] + 10 * state[7]
        if not self.copilot:
            shaping -= 100 * np.hypot(dx, state[1])
        reward = 0 if self.prev_shaping is None else shaping - self.prev_shaping
        self.prev_shaping = shaping
        reward -= m_power * 0.30
        reward -= s_power * 0.03

        # Empowerment
        if self.empowerment_coeff > 0 and not self.fake_step:
            emp = self.compute_empowerment(state, OBS_DIM)
            reward += self.empowerment_coeff * emp

        # ======== 结束判定 ========
        timeout = self.curr_step >= MAX_NUM_STEPS
        at_site = (any(x1 <= pos.x <= x2 for (x1, x2) in self.helipads)
                   and self.legs[0].ground_contact and self.legs[1].ground_contact)

        self.num_steps_at_site = self.num_steps_at_site + 1 if at_site else 0
        done = (self.game_over or abs(state[0]) >= 1.0 or timeout
                or not self.lander.awake or self.num_steps_at_site > 3)

        info = {}

        # ---------- belief 更新 & 注入 ----------
        if not self.fake_step:
            from utils import env_utils as utils
            self._update_belief(utils.human_agent_action)
        info['belief'] = self.belief.copy()
        info['true_goal'] = self.true_goal
        # ---------------------------------------

        info['pad_centers'] = centers

        if done and not self.fake_step:
            reward = -100 if (self.game_over or abs(state[0]) >= 1.0 or timeout) else +100
            print(reward)

            info['trajectory'], info['actions'] = self.trajectory, self.actions

            # ---- CSV 日志 ----
            trajectory_np = np.asarray(self.trajectory)
            dist_to_goal = np.sqrt(trajectory_np[:, 0] ** 2 + trajectory_np[:, 1] ** 2)
            speed = np.sqrt(trajectory_np[:, 2] ** 2 + trajectory_np[:, 3] ** 2)
            angle = trajectory_np[:, 4]

            with open(self.log_file + '_dist.csv', 'a', newline='') as f:
                csv.writer(f).writerow([reward] + list(dist_to_goal))
            with open(self.log_file + '_speed.csv', 'a', newline='') as f:
                csv.writer(f).writerow([reward] + list(speed))
            with open(self.log_file + '_angle.csv', 'a', newline='') as f:
                csv.writer(f).writerow([reward] + list(angle))

        # ---------- Copilot 拼接人类动作 ----------
        if self.copilot and not self.fake_step:
            if self.pilot_is_human:
                pilot_action = onehot_encode(self.pilot_policy(state[None, :]))
            else:
                pilot_action = onehot_encode(self.pilot_policy.step(state[None, :]))
            self.past_pilot_actions[ACT_DIM:] = self.past_pilot_actions[:-ACT_DIM]
            self.past_pilot_actions[:ACT_DIM] = pilot_action
            state = np.concatenate((state, self.past_pilot_actions))

        return state, reward, done, info

    # ------------------------------------------------------------------
    # 渲染
    # ------------------------------------------------------------------
    def render(self, mode='human', close=False):
        if close:
            self.close()

        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.Viewer(VIEWPORT_W, VIEWPORT_H)
            self.viewer.set_bounds(0, VIEWPORT_W / SCALE, 0, VIEWPORT_H / SCALE)

        # 粒子衰减
        for obj in self.particles:
            obj.ttl -= 0.15
            obj.color1 = (max(0.2, 0.2 + obj.ttl),
                          max(0.2, 0.5 * obj.ttl),
                          max(0.2, 0.5 * obj.ttl))
            obj.color2 = obj.color1
        self._clean_particles(False)

        # 地形背景
        for p in self.sky_polys:
            self.viewer.draw_polygon(p, color=(0, 0, 0))

        # 飞船 + 粒子
        for obj in self.particles + self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                if isinstance(f.shape, circleShape):
                    t = rendering.Transform(translation=trans * f.shape.pos)
                    self.viewer.draw_circle(f.shape.radius, 20,
                                            color=obj.color1).add_attr(t)
                    self.viewer.draw_circle(f.shape.radius, 20,
                                            color=obj.color2,
                                            filled=False, linewidth=2).add_attr(t)
                else:
                    path = [trans * v for v in f.shape.vertices]
                    self.viewer.draw_polygon(path, color=obj.color1)
                    path.append(path[0])
                    self.viewer.draw_polyline(path, color=obj.color2, linewidth=2)

        # 跑道旗杆（左黄右蓝）
        flag_colors = [(0.8, 0.8, 0.0), (0.0, 0.6, 0.9)]
        for idx, (x1, x2) in enumerate(self.helipads):
            fcolor = flag_colors[idx % len(flag_colors)]
            for x in (x1, x2):
                flagy1, flagy2 = self.helipad_y, self.helipad_y + 50 / SCALE
                self.viewer.draw_polyline([(x, flagy1), (x, flagy2)],
                                          color=(1, 1, 1))
                self.viewer.draw_polygon(
                    [(x, flagy2),
                     (x, flagy2 - 10 / SCALE),
                     (x + 25 / SCALE, flagy2 - 5 / SCALE)],
                    color=fcolor)

        return self.viewer.render(return_rgb_array=(mode == 'rgb_array'))

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    # ------------------------------------------------------------------
    # Empowerment 近似
    # ------------------------------------------------------------------
    def compute_empowerment(self, state, state_dim,
                            horizon=5, n_traj=10):
        """方差近似 empowerment"""
        X = np.zeros((n_traj, state_dim))
        self.fake_step = True

        # 备份飞船与粒子状态
        lander_pos = self.lander.position.copy()
        lander_angle = self.lander.angle
        lander_lin_vel = self.lander.linearVelocity.copy()
        lander_ang_vel = self.lander.angularVelocity

        leg_pos = [leg.position.copy() for leg in self.legs]
        particle_pos = [p.position.copy() for p in self.particles]

        bak_curr_step = self.curr_step
        bak_shaping = self.prev_shaping
        bak_game_over = self.game_over
        bak_num_site = self.num_steps_at_site

        for n in range(n_traj):
            for _ in range(horizon):
                a = self.action_space.sample()
                x, _, done, _ = self.step(a)
                if done:
                    break
            X[n, :] = x

            # 还原
            for leg, ppos in zip(self.legs, leg_pos):
                leg.position = ppos
            for p, pos in zip(self.particles, particle_pos):
                p.position = pos
            self.lander.position = lander_pos
            self.lander.angle = lander_angle
            self.lander.linearVelocity = lander_lin_vel
            self.lander.angularVelocity = lander_ang_vel
            self.curr_step = bak_curr_step
            self.prev_shaping = bak_shaping
            self.game_over = bak_game_over
            self.num_steps_at_site = bak_num_site

        est_emp = float(np.var(X[:, :2])) if X.size else 0.0
        self.fake_step = False
        return est_emp
