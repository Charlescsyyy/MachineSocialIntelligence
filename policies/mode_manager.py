import numpy as np

class ModeManager:
    def __init__(self, switch_thresh=0.85, blend_steps=60):
        self.mode = "SHARED"
        self.goal_id = None
        self.switch_thresh = switch_thresh
        self.blend_steps = blend_steps
        self._alpha = 1.0          # 1=全玩家, 0=全auto
        self._frames_since_switch = 0

    def update(self, belief):
        """根据 belief 判断是否要切到 AUTO 模式"""
        if self.mode == "SHARED" and belief.max() > self.switch_thresh:
            self.mode = "AUTO"
            self.goal_id = int(belief.argmax())
            self._alpha = 1.0
            self._frames_since_switch = 0

    def blend_action(self, a_human, a_copilot, a_auto):
        """
        根据当前模式输出最终动作。
        * SHARED : players vs copilot (Emp) 直接用论文里的混控规则
        * AUTO   : 前 blend_steps 帧做线性插值，之后完全用 autopilot
        """
        if self.mode == "SHARED":
            # 论文原式：若动作冲突且copilot自信度高则overrule
            return a_copilot
        else:  # AUTO
            if self._frames_since_switch < self.blend_steps:
                self._alpha = max(0.0, 1 - self._frames_since_switch / self.blend_steps)
                self._frames_since_switch += 1
                return self._alpha * a_copilot + (1 - self._alpha) * a_auto
            else:
                return a_auto


    def reset(self):
        """在 episode 结束时清零所有内部状态"""
        self.mode = "SHARED"
        self.goal_id = None
        self._alpha = 1.0
        self._frames_since_switch = 0
