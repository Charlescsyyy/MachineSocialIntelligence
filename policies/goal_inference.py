"""
Minimal Goal‑Inference module for 2‑pad Lunar‑Lander (GOMA‑style level‑1 ToM).

* Maintains a belief vector  b_t = [p_left, p_right]
* 观测 = (state, pilot_action)
    state        : 9‑D physics state (不含拼接的人类动作)
    pilot_action : 连续动作 a_h = [main, side]  or  离散 one‑hot 已转成 [-1,0,1] 形式
* Likelihood    :  启发式基于 side thruster 方向
                  P(u | g=left)  = 0.8 if side<0 else 0.2
                  P(u | g=right) = 0.8 if side>0 else 0.2
* Belief update :  b_t(g) ∝ P(u|g) · b_{t‑1}(g)

Later you can replace `likelihood()` with a small neural net that models
P(u|g,s) learned from human data.
"""

import numpy as np

LEFT, RIGHT = 0, 1


class GoalInference:
    """
    Parameters
    ----------
    pad_centers : list[float]
        x‑coordinate of each pad's centre, e.g.  [-0.25, 0.25]  (world frame)
    beta : float
        高似然概率，默认 0.8；低似然 = 1‑beta
    """

    def __init__(self, pad_centers, beta: float = 0.8, init_belief=None):
        self.pad_centers = pad_centers
        self.beta = np.clip(beta, 0.5, 1.0)      # keep valid
        self.reset(init_belief)

    # ------------------------------------------------------------------ #
    # public API                                                         #
    # ------------------------------------------------------------------ #
    def reset(self, init_belief=None):
        """Call on env.reset() to clear history."""
        if init_belief is None:
            init_belief = np.ones(2) / 2.0
        self.belief = np.asarray(init_belief, dtype=np.float32)

    def update(self, state, pilot_action):
        """
        Update belief and return it.

        Parameters
        ----------
        state : np.ndarray, shape (9,)
        pilot_action : np.ndarray, shape (2,)  # [main, side] ∈ [-1,1]

        Returns
        -------
        belief : np.ndarray, shape (2,)   # sums to 1
        """
        # make sure we handle python list etc.
        u = np.asarray(pilot_action, dtype=np.float32)

        for g in (LEFT, RIGHT):
            self.belief[g] *= self.likelihood(u, g)

        # normalise & avoid divide‑by‑0
        s = self.belief.sum()
        if s < 1e-8:
            self.belief[:] = 0.5
        else:
            self.belief /= s
        return self.belief

    # ---------------- helper getters ---------------------------------- #
    def most_likely(self) -> int:
        """Return argmax of current belief (0:left, 1:right)."""
        return int(self.belief.argmax())

    def confidence(self) -> float:
        """Return max probability of current belief."""
        return float(self.belief.max())

    def entropy(self) -> float:
        """Return Shannon entropy of belief (base‑2)."""
        p = np.clip(self.belief, 1e-9, 1.0)
        return float(-(p * np.log2(p)).sum())

    # ------------------------------------------------------------------ #
    # internal                                                           #
    # ------------------------------------------------------------------ #
    def likelihood(self, pilot_action: np.ndarray, goal_id: int) -> float:
        """
        Return P(u | g).  Simple heuristic using side‑thruster sign.

        If player fires left thruster (`side < 0`) we treat it as evidence
        for wanting the **left** pad; likewise for right thruster.
        """
        side = np.sign(pilot_action[1])  # -1,0,+1
        if goal_id == LEFT:
            return self.beta if side < 0 else (1.0 - self.beta)
        else:  # RIGHT
            return self.beta if side > 0 else (1.0 - self.beta)

    # nice `repr` for debug
    def __repr__(self):
        return f"<GoalInference belief={self.belief}>"
