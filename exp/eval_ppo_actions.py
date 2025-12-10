# exp/eval_ppo_actions.py

from pathlib import Path
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from include.params import create_default_lip_params
from rl_src.foot_residual_env import LIPFootResidualEnv, EnvConfig


def make_env(seed: int):
    def _init():
        params = create_default_lip_params()
        env_cfg = EnvConfig()
        env = LIPFootResidualEnv(params=params, env_config=env_cfg)
        env.reset()
        return env
    return _init


def main():
    project_root = Path(__file__).resolve().parents[1]
    runs_root = project_root / "exp" / "ppo_foot_residual"

    seed = 15
    run_dir = runs_root / f"seed_{seed:03d}"
    model_path = run_dir / "checkpoint_2240000_steps.zip"   # adjust if needed

    print(f"Loading model from {model_path}")
    model = PPO.load(str(model_path))

    env = DummyVecEnv([make_env(seed + 100)])
    obs = env.reset()

    actions = []

    num_episodes = 1000
    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        ep_steps = 0

        while not done and ep_steps < 100:  # safety cap
            action, _ = model.predict(obs, deterministic=False)
            action /= 100
            actions.append(action.copy())
            obs, rewards, dones, infos = env.step(action)
            done = bool(dones[0])
            ep_steps += 1

    actions = np.squeeze(np.array(actions))  # shape (~N, 2)
    print(f"Collected {actions.shape[0]} actions")
    print(f"Average Episode Length: {actions.shape[0]/num_episodes}")
    print("Mean action magnitude:", np.abs(actions).mean(axis=0))
    print("Std  action:", actions.std(axis=0))
    print("First 10 actions:\n", actions[:10])


if __name__ == "__main__":
    main()
