#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Human-in-the-loop Empowerment Lander
-----------------------------------
保持原版手感 + 轨迹录制 + 可选 RL 目标推断器
"""

from __future__ import division
import os, sys, time, argparse
from collections import Counter
from typing import Optional

import numpy as np
from pyglet.window import key as pygkey

# ---------- 全局参数 ----------
K_GOALS = 2       # 跑道数量，应与 LunarLanderEmpowerment.k_goals 保持一致
MAX_EP_LEN = 500
N_TRAIN_EP  = 50
# --------------------------------

# 把项目根加入搜索路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from envs.lunar_lander_emp import LunarLanderEmpowerment
from policies import CoPilotPolicy
from baselines import logger

try:
    from src.goal_predictor import GoalPredictor
except ImportError:
    GoalPredictor = None

from src.record_wrapper import TrajectoryRecorder

import utils.env_utils as utils
from utils.env_utils import init_human_action, encode_human_action

LEFT, RIGHT, UP, DOWN = pygkey.LEFT, pygkey.RIGHT, pygkey.UP, pygkey.DOWN
utils.human_agent_active = False


# ------------------ 键盘回调 ------------------
def key_press(key, mod):
    a = int(key)
    if a == LEFT:
        utils.human_agent_action[1] = 0
    elif a == RIGHT:
        utils.human_agent_action[1] = 2
    elif a == UP:
        utils.human_agent_action[0] = 1
    elif a == DOWN:
        utils.human_agent_action[0] = 0
    utils.human_agent_active = True


def key_release(key, mod):
    a = int(key)
    if a in (LEFT, RIGHT):
        utils.human_agent_action[1] = 1
    elif a in (UP, DOWN):
        utils.human_agent_action[0] = 0
    utils.human_agent_active = False


def human_pilot_policy(_obs):
    return encode_human_action(utils.human_agent_action)


# ------------------ 核心实验 ------------------
def run_test(base_dir: str,
             empowerment: float,
             scope: str,
             record_traj: bool,
             goal_rl_path: Optional[str],
             pilot_tol: float,
             k_goals: int):
    utils.human_agent_action = init_human_action()
    utils.human_agent_active = False

    # ---------- 可选加载 RL 目标推断器 ----------
    goal_predictor = None
    if goal_rl_path and GoalPredictor is not None:
        print(f"[Info] Loading GoalPredictor from {goal_rl_path}")
        goal_predictor = GoalPredictor(goal_rl_path, k_goals=k_goals)

    # ---------- 构造底层环境 ----------
    base_env = LunarLanderEmpowerment(
        empowerment=empowerment,
        ac_continuous=False,
        pilot_policy=human_pilot_policy,
        pilot_is_human=True,
        log_file=os.path.join(base_dir, scope),
        goal_predictor=goal_predictor,
        k_goals=k_goals
    )

    # ---------- 装轨迹记录壳 ----------
    env = (TrajectoryRecorder(base_env,
                              csv_path=os.path.join(base_dir,
                                                    f"{scope}_traj.csv"))
           if record_traj else base_env)

    # ---------- 人机交互 ----------
    env.render()
    env.unwrapped.viewer.window.on_key_press = key_press
    env.unwrapped.viewer.window.on_key_release = key_release

    # ---------- Copilot ----------
    try:
        copilot_policy = CoPilotPolicy(
            base_dir,
            policy_path=f"policies/pretrained_policies/{scope}_policy.pkl")
    except FileNotFoundError:
        copilot_policy = CoPilotPolicy(base_dir)
        print("[Info] No pretrained CoPilot, training from scratch.")

    copilot_policy.learn(
        env,
        max_timesteps=MAX_EP_LEN * N_TRAIN_EP,
        pilot=human_pilot_policy,
        pilot_is_human=True,
        pilot_tol=pilot_tol,
        copilot_scope=scope
    )
    env.close()

    # ---------- 结果统计 ----------
    rew = copilot_policy.reward_data
    mean_rewards = float(np.mean(rew["rewards"]))
    outcome = [r if r % 100 == 0 else 0 for r in rew["outcomes"]]
    outcome_distrns = Counter(outcome)

    with open(os.path.join(base_dir, f"result_{scope}.txt"), "w") as f:
        f.write(f"Empowerment: {empowerment}\n")
        f.write(f"Mean reward: {mean_rewards}\n")
        f.write(f"Outcome distribution: {outcome_distrns}\n")


# ------------------ 实验入口 ------------------
def run_experiment(empowerment: bool,
                   record: bool,
                   goal_rl_path: Optional[str],
                   pilot_tol: float,
                   k_goals: int):
    base_dir = os.path.join(os.getcwd(), "data", "human_co")
    logger.configure(dir=base_dir,
                     format_strs=["stdout", "log", "csv", "tensorboard"])
    time.sleep(1)
    start = time.time()

    run_test(base_dir,
             empowerment=0.001 if empowerment else 0.0,
             scope="emp" if empowerment else "noemp",
             record_traj=record,
             goal_rl_path=goal_rl_path,
             pilot_tol=pilot_tol,
             k_goals=k_goals)

    print(f"{time.time() - start:.1f}s Total time taken")


# ------------------ CLI ------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Empowerment Lander Testbed")
    parser.add_argument("--empowerment", action="store_true",
                        help="启用 empowerment 场景")
    parser.add_argument("--record", action="store_true",
                        help="将 belief/human_action/true_goal 写入 CSV")
    parser.add_argument("--goal_rl", type=str, default=None,
                        help="训练好的 Goal-RL 权重路径（不带扩展名）")
    parser.add_argument("--pilot_tol", type=float, default=0.8,
                        help="人-机动作融合阈值")
    parser.add_argument("--k_goals", type=int, default=K_GOALS,
                        help="可能目标数量；需与其他脚本一致")
    args = parser.parse_args()

    run_experiment(empowerment=args.empowerment,
                   record=args.record,
                   goal_rl_path=args.goal_rl,
                   pilot_tol=args.pilot_tol,
                   k_goals=args.k_goals)
