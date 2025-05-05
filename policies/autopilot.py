# import numpy as np

# class SimpleAutopilot:
#     def __init__(self, pad_centers):
#         self.pad_centers = pad_centers

#     def step(self, obs9, goal_id):
#         """
#         obs9 = env 返回的前 9 维物理状态
#         返回连续动作 [main, side] ∈ [-1,1]^2
#         """
#         x, y, vx, vy, angle, ang_vel, leg0, leg1, rel_pad = obs9
#         target_x = (self.pad_centers[goal_id] - 0) / (VIEWPORT_W / SCALE / 2)

#         # 横向 PD
#         side_cmd = np.clip( 5.0 * (x - target_x) + 2.0 * vx, -1, 1 )

#         # 垂直 + 姿态：高度>某阈值用 60% 主推，越低越收油
#         main_cmd = 0.6 if y > -0.1 else 0.3 if y > -0.2 else 0.0
#         return np.array([main_cmd, side_cmd], dtype=np.float32)
from envs.lunar_lander_emp import VIEWPORT_W, SCALE
import numpy as np

class SimpleAutopilot:
    def __init__(self, pad_centers):
        self.pad_centers = pad_centers    # 形如 [‑0.25, 0.23] (world 坐标)

    def step(self, obs9, goal_id):
        x, y, vx, vy, angle, ang_vel, *_ = obs9

        # pad_centers 是 world‑x，需要先归一化成和 obs9[0] 同一量纲
        target_x = (self.pad_centers[goal_id] - VIEWPORT_W / SCALE / 2) / (VIEWPORT_W / SCALE / 2)

        side_cmd = np.clip( 5.0*(x - target_x) + 2.0*vx, -1, 1 )
        main_cmd = 0.6 if y > -0.1 else (0.3 if y > -0.2 else 0.0)
        return np.array([main_cmd, side_cmd], dtype=np.float32)