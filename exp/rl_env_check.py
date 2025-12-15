# exp/sanity_check_env.py

import numpy as np
from rl_src.foot_residual_env import LIPFootResidualEnv
from include.params import create_default_lip_params


def main():
    # --- Create params + env ---
    params = create_default_lip_params()
    env = LIPFootResidualEnv(params=params)

    print("Action space:", env.action_space)
    print("Observation space:", env.observation_space)

    # --- Reset ---
    obs, info = env.reset(seed=0)
    print("\nInitial observation:")
    print(obs)
    print("Obs shape:", obs.shape)
    print("Info keys:", info.keys())

    # --- Step through a few random actions ---
    print("\nStepping through env with random actions...")
    for t in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        print(f"\nStep {t}:")
        print("  action =", action)
        print("  obs.shape =", obs.shape)
        print("  reward =", reward)
        print("  terminated =", terminated, " truncated =", truncated)
        print("  info:", info)

        if terminated or truncated:
            print("\nEpisode ended early.")
            break

    print("\nSanity check complete — if no errors occurred, the env is functional.")


if __name__ == "__main__":
    main()
