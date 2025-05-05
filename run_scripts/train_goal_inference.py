#!/usr/bin/env python
# -----------------------------------------------------------
# 训练 Goal-Inference 策略（PPO2）
# -----------------------------------------------------------

import os, sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import joblib
import pandas as pd

from baselines import logger
from baselines.ppo2 import ppo2
from baselines.common.vec_env import DummyVecEnv, VecMonitor  # 修改处

from envs.lunar_lander_emp import LunarLanderEmpowerment
from src.goal_inference_env import GoalInferenceEnv
import utils.env_utils as utils
utils.human_agent_action = [0, 1]    # 主引擎=0, 侧推中立=1


def main(args):
    base_env = LunarLanderEmpowerment(
        empowerment=0.001,
        ac_continuous=False,
        pilot_policy=None,
        pilot_is_human=False,
        log_file='tmp'
    )

    single_env = GoalInferenceEnv(base_env, k_goals=2)
    vec_env    = DummyVecEnv([lambda: single_env])
    env        = VecMonitor(vec_env, logger.get_dir())        # 修改处

    policy = ppo2.learn(
        network='mlp',
        env=env,
        total_timesteps=args.steps,
        nsteps=128,
        nminibatches=32
    )

    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    policy.save(args.save) 
    print(f"Saved Goal-RL weights to: {args.save}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Goal-Inference PPO2')
    parser.add_argument('--steps', type=int, default=2_000_000)
    parser.add_argument('--save', type=str, default='policies/goal_rl.pkl')
    args = parser.parse_args()

    logger.configure(dir='goal_rl_log',
                     format_strs=['stdout', 'log', 'csv', 'tensorboard'])

    main(args)