from pathlib import Path
import numpy as np

from stable_baselines3 import PPO
from include.params import create_default_lip_params
from rl_src.foot_residual_env import LIPFootResidualEnv, EnvConfig

# --- Paths ---
project_root = Path(__file__).resolve().parents[1]  # adjust if running elsewhere
run_dir = project_root / "exp" / "ppo_foot_residual" / "seed_000"  # change seed
model_path = run_dir / "final_model.zip"

# --- Recreate env (must match training) ---
params = create_default_lip_params()
env_cfg = EnvConfig()
env = LIPFootResidualEnv(params=params, env_config=env_cfg)

# --- Load model ---
model = PPO.load(str(model_path), env=env)

# --- Run a few evaluation episodes ---
n_episodes = 5
for ep in range(n_episodes):
    obs, info = env.reset()
    done = False
    truncated = False
    ep_reward = 0.0
    steps = 0

    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        ep_reward += reward
        steps += 1

    print(f"Episode {ep}: return={ep_reward:.3f}, steps={steps}, fell={info['fell']}")
