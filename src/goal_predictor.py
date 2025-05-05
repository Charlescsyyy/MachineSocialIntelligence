# src/goal_predictor.py
import joblib
import numpy as np

class GoalPredictor:
    """
    轻量封装：加载 Baselines PPO2 policy，并暴露 predict(obs)→goal_id
    """
    def __init__(self, ckpt_path: str, k_goals: int = 2):
        # 直接反序列化 policy 对象
        self.policy = joblib.load(ckpt_path)
        self.k_goals = k_goals

    def predict(self, obs: np.ndarray) -> int:
        """
        Parameters
        ----------
        obs : np.ndarray, shape=(obs_dim,)
        Returns
        -------
        goal_id : int  0..k_goals-1
        """
        # baselines policy.step() 返回 (actions, values, states, neglogpacs)
        act, *_ = self.policy.step(obs[None, :], deterministic=True)
        return int(act.squeeze())

    # 兼容 human_emp 中的调用写法
    def __call__(self, obs: np.ndarray) -> int:
        return self.predict(obs)