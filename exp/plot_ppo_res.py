# exp/plot_episode_lengths.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def load_monitor_csv(path: str | Path) -> pd.DataFrame:
    """
    Load a stable-baselines3 Monitor CSV.
    Lines starting with '#' are comments and will be ignored.
    """
    path = Path(path)
    df = pd.read_csv(path, comment="#")
    # Expected columns: 'r' (reward), 'l' (ep_len), 't' (time)
    return df


def plot_episode_lengths(
    csv_path: str | Path,
    window: int = 50,
    title: str | None = None,
):
    df = load_monitor_csv(csv_path)

    # Episode length (number of env steps per episode)
    ep_len = df["l"].to_numpy(dtype=float)

    # Cumulative timesteps (x-axis)
    cum_timesteps = np.cumsum(ep_len)

    # Moving average of episode length (y-axis, smoothed)
    ep_len_series = pd.Series(ep_len)
    ep_len_ma = ep_len_series.rolling(window=window, min_periods=1).mean().to_numpy()

    plt.figure(figsize=(8, 5))

    # Raw episode lengths (optional: faint scatter)
    # plt.scatter(
    #     cum_timesteps,
    #     ep_len,
    #     s=8,
    #     alpha=0.3,
    #     label="Episode length (raw)",
    # )

    N = len(cum_timesteps)
    half = N // 2

    cum_timesteps = cum_timesteps[:half]
    ep_len_ma = ep_len_ma[:half]

    # Smoothed curve
    plt.plot(
        cum_timesteps,
        ep_len_ma,
        linewidth=2,
        
    )

    plt.xlabel("Cumulative environment timesteps")
    plt.ylabel("Episode length (steps)")
    if title is None:
        title = f"Episode length vs timesteps"
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage: point this to your monitor CSV
    csv_file = "exp/ppo_foot_residual/seed_007/train_logs/monitor_seed7.monitor.csv"
    plot_episode_lengths(csv_file, window=50)
