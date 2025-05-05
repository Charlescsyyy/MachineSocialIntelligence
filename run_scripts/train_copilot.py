from __future__ import division
import os, random, math, uuid, time, types, argparse, multiprocessing
from copy import copy
from collections import defaultdict, Counter

import numpy as np
import gym
from gym import spaces, wrappers
from gym.envs.registration import register

from envs import LunarLanderEmpowerment, LunarLander
from policies import (FullPilotPolicy, LaggyPilotPolicy, NoopPilotPolicy,
                      NoisyPilotPolicy, SensorPilotPolicy, CoPilotPolicy)

from baselines import logger
from baselines.common.schedules import LinearSchedule
from baselines import deepq
from baselines.common import models
from baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from baselines.deepq.deepq import ActWrapper
from baselines.common.tf_util import make_session

import cloudpickle
import tensorflow as tf
from matplotlib import pyplot as plt
from datetime import datetime

# ---------- 可选 doodad 依赖 ----------
from importlib import util as _imp_util
if _imp_util.find_spec('doodad.easy_sweep') is not None:
    import doodad as dd
    import doodad.mount as mount
    import doodad.easy_sweep.launcher as launcher
    from doodad.easy_sweep.hyper_sweep import run_sweep_doodad
else:
    dd = mount = launcher = run_sweep_doodad = None
# -------------------------------------

from experiment_utils import config
from experiment_utils.utils import query_yes_no

EXP_NAME = "CopilotTraining"


def run_ep(policy, env, max_ep_len, render=False):
    """评估独立 pilot（不含 copilot）"""
    obs, done, totalr = env.reset(), False, 0.0
    for _ in range(max_ep_len):
        if done:
            break
        action = policy.step(obs[None, :])
        obs, r, done, info = env.step(action)
        if render:
            env.render()
        totalr += r
    # 任务成功：奖励是 ±100；否则 0
    outcome = r if r % 100 == 0 else 0
    return totalr, outcome


def run_ep_copilot(policy, env, max_ep_len, pilot, pilot_tol, render=False):
    """评估 pilot + copilot 协作"""
    obs, done, totalr = env.reset(), False, 0.0
    pilot_actions = np.zeros((env.num_concat * env.act_dim))
    for _ in range(max_ep_len):
        if done:
            break
        action, pilot_actions = policy.step(obs[None, :],
                                           pilot, pilot_tol, pilot_actions)
        obs, r, done, info = env.step(action)
        if render:
            env.render()
        totalr += r
    outcome = r if r % 100 == 0 else 0
    return totalr, outcome


def run_experiment(empowerment, exp_title, seed):
    """一轮完整训练 + 交叉评估"""
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)

    base_dir = os.path.join(config.DOCKER_MOUNT_DIR, EXP_NAME)
    logger.configure(dir=base_dir,
                     format_strs=['stdout', 'log', 'csv', 'tensorboard'])

    # 记录超参数
    with open(os.path.join(base_dir, "config.txt"), "w") as f:
        f.write(f"Empowerment: {empowerment}\nNum concat: 20\nSeed: {seed}\n")

    max_ep_len = 1000
    n_training_episodes = 500
    max_timesteps = max_ep_len * n_training_episodes

    # === 1. 训练 Full Pilot（专家）===
    env = LunarLanderEmpowerment(empowerment=0.0, ac_continuous=False)
    full_pilot = FullPilotPolicy(base_dir)
    full_pilot.learn(env, max_timesteps)

    # === 2. 构造四类缺陷 pilot ===
    laggy_pilot   = LaggyPilotPolicy(base_dir,   full_policy=full_pilot.policy)
    noisy_pilot   = NoisyPilotPolicy(base_dir,   full_policy=full_pilot.policy)
    noop_pilot    = NoopPilotPolicy(base_dir,    full_policy=full_pilot.policy)
    sensor_pilot  = SensorPilotPolicy(base_dir,  full_policy=full_pilot.policy)
    sim_pilots    = [full_pilot, laggy_pilot, noisy_pilot, noop_pilot, sensor_pilot]
    pilot_names   = ['full', 'laggy', 'noisy', 'noop', 'sensor']
    pilot_tol_map = {'noop': 0, 'laggy': 0.7, 'noisy': 0.3, 'sensor': 0.1}

    # === 3. 训练各 pilot 对应的 Copilot ===
    copilot_of_pilot = {}
    for pid, ptol in pilot_tol_map.items():
        train_pilot = eval(f"{pid}_pilot")
        cfg = dict(pilot_policy=train_pilot, pilot_tol=ptol,
                   reuse=True, copilot_scope=f"co_deepq_{pid}")
        co_env = LunarLanderEmpowerment(empowerment=empowerment,
                                        ac_continuous=False, **cfg)
        copilot = CoPilotPolicy(base_dir)
        copilot.learn(co_env, max_timesteps=max_timesteps, **cfg)
        copilot_of_pilot[pid] = copilot

    # === 4. 交叉评估 ===
    n_eval_eps = 100
    cross_evals = {}
    for train_pid, ptol in pilot_tol_map.items():
        train_copilot = copilot_of_pilot[train_pid]
        for eval_pid, eval_ptol in pilot_tol_map.items():
            eval_pilot = eval(f"{eval_pid}_pilot")
            eval_env = LunarLanderEmpowerment(empowerment=0,
                                              ac_continuous=False,
                                              pilot_policy=eval_pilot)
            rewards, outcomes = [], []
            for _ in range(n_eval_eps):
                r, o = run_ep_copilot(train_copilot, eval_env,
                                      max_ep_len, eval_pilot, eval_ptol)
                rewards.append(r); outcomes.append(o)
            cross_evals[(train_pid, eval_pid)] = (np.mean(rewards),
                                                 Counter(outcomes))

    # 保存评估结果
    with open(os.path.join(base_dir, "cross_eval.txt"), "w") as f:
        for (tr, ev), (mr, dist) in cross_evals.items():
            f.write(f"Train: {tr}, Eval: {ev} | MeanR: {mr:.1f} "
                    f"| Outcomes: {dict(dist)}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Empowerment Lander Copilot Trainer")
    parser.add_argument('--exp_title',   type=str, default='')
    parser.add_argument('--mode',        type=str, default='local',
                        choices=['local', 'local_docker', 'ec2'])
    parser.add_argument('--empowerment', type=float, default=0.001)
    parser.add_argument('--seed',        type=int,   default=1)
    args = parser.parse_args()

    # ---------- 只有需要远程 / Docker 时才构建 sweeper ----------
    if args.mode in ['ec2', 'local_docker']:
        if launcher is None:
            raise ImportError("当前环境未安装 doodad，无法使用 "
                              f"'{args.mode}' 模式")
        local_mount = mount.MountLocal(local_dir=config.BASE_DIR,
                                       pythonpath=True)
        docker_mount = os.path.join(config.DOCKER_MOUNT_DIR, EXP_NAME)
        sweeper = launcher.DoodadSweeper(
            [local_mount],
            docker_img=config.DOCKER_IMAGE,
            docker_output_dir=docker_mount,
            local_output_dir=os.path.join(config.DATA_DIR, 'local', EXP_NAME)
        )
        sweeper.mount_out_s3 = mount.MountS3(s3_path='',
                                             mount_point=docker_mount,
                                             output=True)

    # ---------- 根据 mode 选择执行 ----------
    if args.mode == 'ec2':
        if query_yes_no("Launch jobs on EC2?"):
            sweeper.run_sweep_ec2(
                run_experiment,
                {'empowerment': [args.empowerment],
                 'exp_title'   : [args.exp_title],
                 'seed'        : [args.seed]},
                bucket_name=config.S3_BUCKET_NAME,
                instance_type='c4.2xlarge',
                region='us-west-1',
                s3_log_name=EXP_NAME,
                add_date_to_logname=True)
    elif args.mode == 'local_docker':
        run_sweep_doodad(
            run_experiment,
            {'empowerment': [args.empowerment]},
            run_mode=dd.mode.LocalDocker(image=sweeper.image),
            mounts=sweeper.mounts)
    else:  # --mode local
        run_experiment(empowerment=args.empowerment,
                       exp_title=args.exp_title or 'local',
                       seed=args.seed)