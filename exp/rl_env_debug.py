import numpy as np
from include.params import create_default_lip_params
from rl_src.foot_residual_env import LIPFootResidualEnv, EnvConfig

def run_zero_policy(n_episodes=1000):
    params = create_default_lip_params()
    env_cfg = EnvConfig()
    env = LIPFootResidualEnv(params=params, env_config=env_cfg)

    returns = []
    lengths = []
    falls = 0

    for ep in range(n_episodes):
        obs, info = env.reset(seed=ep)  # different seeds for noise
        done = False
        truncated = False
        total_r = 0.0
        steps = 0

        while not (done or truncated):
            action = np.zeros(2, dtype=np.float32)  # NO residuals
            obs, reward, done, truncated, info = env.step(action)
            total_r += reward
            steps += 1

        returns.append(total_r)
        lengths.append(steps)
        if info.get("fell", False):
            falls += 1

        #print(f"Ep {ep}: R={total_r:.3f}, steps={steps}, fell={info.get('fell', False)}")

    print("\nZERO policy summary:")
    print(f"  mean return: {np.mean(returns):.3f}")
    print(f"  mean ep len: {np.mean(lengths):.2f}")
    print(f"  falls: {falls}/{n_episodes}")

if __name__ == "__main__":
    run_zero_policy()
