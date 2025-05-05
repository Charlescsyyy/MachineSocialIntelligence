# src/record_wrapper.py
import gym, csv, os, numpy as np

class TrajectoryRecorder(gym.Wrapper):
    """
    只做一件事：在 step() 时把 (belief, human_action, true_goal) 写 CSV
    不改 observation、reward、done。
    """
    def __init__(self, env: gym.Env, csv_path: str):
        super().__init__(env)
        self.csv_path = csv_path
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        self.csv_f = open(csv_path, 'w', newline='')
        self.writer = csv.writer(self.csv_f)
        head = ['t'] + [f'bel{i}' for i in range(4)] + ['human0', 'human1', 'true_goal']
        self.writer.writerow(head)
        self.t = 0

    def step(self, action):
        obs, rew, done, info = self.env.step(action)

        # belief / true_goal 由内层 env 决定；若没有写入则默认 0 / -1
        belief = info.get('belief', np.zeros(4))
        true_goal = info.get('true_goal', -1)
        human0, human1 = getattr(self.env, 'last_human_action', (0.0, 1.0))

        self.writer.writerow(
            [self.t] + list(belief) + [human0, human1, true_goal])
        self.t += 1
        if done:
            self.csv_f.flush()
            self.t = 0
        return obs, rew, done, info

    def close(self):
        self.csv_f.close()
        super().close()