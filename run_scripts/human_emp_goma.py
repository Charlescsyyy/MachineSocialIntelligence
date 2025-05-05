# run_scripts/human_emp_goma.py
import time, numpy as np
from pyglet.window import key as pygkey

from envs.lunar_lander_emp import LunarLanderEmpowerment
from policies.goal_inference import GoalInference
from policies.mode_manager   import ModeManager
from policies.autopilot      import SimpleAutopilot
from policies                import CoPilotPolicy      # 项目原生
from utils.env_utils         import disc_to_cont, init_human_action

import utils.env_utils as utils          # 里面放着 encode/历史动作工具

# ---------- 键盘 → 人类动作 ----------
LEFT, RIGHT, UP, DOWN = pygkey.LEFT, pygkey.RIGHT, pygkey.UP, pygkey.DOWN
utils.human_agent_action = init_human_action()         # [main, side]
utils.human_agent_active = False                       # 是否有人类输入

def key_press(k, mod):
    if   k == LEFT:  utils.human_agent_action[1] = 0
    elif k == RIGHT: utils.human_agent_action[1] = 2
    elif k == UP:    utils.human_agent_action[0] = 1
    elif k == DOWN:  utils.human_agent_action[0] = 0
    utils.human_agent_active = True

def key_release(k, mod):
    if k in (LEFT, RIGHT):
        utils.human_agent_action[1] = 1
    elif k in (UP, DOWN):
        utils.human_agent_action[0] = 0
    utils.human_agent_active = False

def encode_human_action(act):
    """把 [main(0/1), side(0/1/2)] 编成 0‑5 的离散动作号"""
    return act[0]*3 + act[1]

def get_keyboard_action():
    """返回连续动作 np.array([main, side])，供 env.step() 直接使用"""
    disc = encode_human_action(utils.human_agent_action)
    return disc_to_cont(disc)

def human_pilot_policy(_obs_batch):
    """给 CopilotPolicy.step() 用的占位 pilot_policy（只会读键盘动作）"""
    return encode_human_action(utils.human_agent_action)


# ---------- 主循环 ----------
def run():
    env = LunarLanderEmpowerment(empowerment=0.001,
                                 ac_continuous=True,
                                 pilot_is_human=True)
    env.render()
    env.unwrapped.viewer.window.on_key_press   = key_press
    env.unwrapped.viewer.window.on_key_release = key_release

    # --- Copilot（Emp‑DQN）
    try:
        copilot = CoPilotPolicy(
            data_dir  ='policies/pretrained_policies',
            policy_path='policies/pretrained_policies/emp_policy.pkl')
        print('[INFO] 预训练 Copilot 已加载')
    except Exception as e:
        print('[WARN] 预训练权重加载失败：', e)
        copilot = CoPilotPolicy(data_dir='policies/pretrained_policies')
        print('[INFO] 将在线学习 Copilot')



    # pad_centers = [(x1 + x2) / 2.0 for (x1, x2) in env.helipads]  # ← list[float]
    # # --- GOMA 组件
    # #gi   = GoalInference(env.helipads, beta=0.8)
    # gi   = GoalInference(pad_centers, beta=0.8)
    # mm   = ModeManager(switch_thresh=0.85)
    # #autop= SimpleAutopilot(env.helipads)
    # autop= SimpleAutopilot(pad_centers)

    # obs  = env.reset()
    # pilot_hist = np.zeros(env.num_concat * 6)      # (NUM_CONCAT × ACT_DIM)
    # ---------- 首次 reset，拿到本局跑道 ----------
    obs = env.reset()
    pad_centers = [(x1 + x2) / 2.0 for (x1, x2) in env.helipads]

# ---------- GOMA 组件 ----------
    gi   = GoalInference(pad_centers, beta=0.8)
    mm   = ModeManager(switch_thresh=0.85)
    autop= SimpleAutopilot(pad_centers)

    pilot_hist = np.zeros(env.num_concat * 6)      # (NUM_CONCAT × ACT_DIM)

    while True:
        # ---------- 人类输入 ----------
        a_pilot = get_keyboard_action()            # 连续 [main, side]

        # ---------- 目标推断 / 模式切换 ----------
        obs9   = obs[:9]
        belief = gi.update(obs9, a_pilot)
        mm.update(belief)                          # 可能从 COPILOT→AUTO

        # ---------- Copilot 动作 ----------
        a_copilot_disc, pilot_hist = copilot.step(
            observation   = obs,
            pilot_policy  = human_pilot_policy,
            pilot_tol     = 0.8,
            pilot_actions = pilot_hist,
            pilot_is_human= True
        )
        a_copilot = disc_to_cont(a_copilot_disc)   # 转成连续

        # ---------- Autopilot（若已对齐目标） ----------
        a_auto = autop.step(obs9, mm.goal_id) if mm.mode == "AUTO" else a_copilot

        # ---------- 动作融合 & 执行 ----------
        action = mm.blend_action(a_pilot, a_copilot, a_auto)
        obs, rew, done, info = env.step(action)
        env.render()

        # ---------- Episode 结束 ----------
        # if done:
        #     gi.reset()
        #     mm.reset()                             # 归零模式 / belief
        #     pilot_hist[:] = 0
        #     utils.human_agent_action = init_human_action()
        #     obs = env.reset()
        #     time.sleep(1)                          # 给玩家一点缓冲时间
        if done:
    # —— 重新开始一局 ——
            obs = env.reset()
            pad_centers = [(x1 + x2) / 2.0 for (x1, x2) in env.helipads]

            gi.reset()
            gi.pad_centers    = pad_centers       # 同步给推断器
            autop.pad_centers = pad_centers       # 同步给自动驾驶
            mm.reset()

            pilot_hist[:] = 0
            utils.human_agent_action = init_human_action()
            time.sleep(1)
            continue                               # 回到循环顶部


if __name__ == "__main__":
    run()