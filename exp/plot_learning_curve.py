# exp/plot_learning_curve.py

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_monitor_csv(path: Path) -> pd.DataFrame:
    """
    Load a single Stable-Baselines3 Monitor CSV.

    The file has comment lines starting with '#', then a header row with:
        r,l,t
    where:
      - r : episode return
      - l : episode length (in env steps)
      - t : time in seconds or steps since the monitor started (per env)

    We keep 'r', 'l', 't' as-is.
    """
    df = pd.read_csv(path, comment="#")
    # Ensure expected columns exist
    required_cols = {"r", "l", "t"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Monitor file {path} missing required columns {required_cols}")
    return df


def load_all_monitors(run_dir: Path) -> pd.DataFrame:
    """
    Find and load all *.monitor.csv files under run_dir (recursively).

    Returns a single concatenated DataFrame with columns:
        - r: episode return
        - l: episode length
        - t: per-env time
    plus:
        - env_id: integer ID for which env/file this row came from
    """
    monitor_files = list(run_dir.rglob("*.monitor.csv"))
    if not monitor_files:
        raise FileNotFoundError(f"No monitor CSV files found in {run_dir}")

    dfs = []
    for env_id, fpath in enumerate(sorted(monitor_files)):
        df = load_monitor_csv(fpath)
        df["env_id"] = env_id
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)

    # Sort episodes by per-env time 't' so earlier episodes across envs
    # are roughly earlier on the global timeline
    df_all = df_all.sort_values(by="t").reset_index(drop=True)
    return df_all


def bin_by_timesteps(
    timesteps: np.ndarray,
    values: np.ndarray,
    bin_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given episode-level timesteps (x) and values (y),
    compute a binned curve over timesteps.

    - timesteps: 1D array, monotonically increasing (e.g., cumsum of episode lengths)
    - values: 1D array of same length (e.g., episode returns or lengths)
    - bin_size: width of timestep bins

    Returns:
        bin_centers, bin_means
    """
    assert timesteps.ndim == 1
    assert values.ndim == 1
    assert timesteps.shape[0] == values.shape[0]

    if len(timesteps) == 0:
        return np.array([]), np.array([])

    max_t = timesteps[-1]
    if max_t <= 0:
        return np.array([]), np.array([])

    # Define bin edges: [0, bin_size), [bin_size, 2*bin_size), ...
    edges = np.arange(0, max_t + bin_size, bin_size)
    bin_centers = []
    bin_means = []

    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        mask = (timesteps >= lo) & (timesteps < hi)
        if not np.any(mask):
            continue
        bin_centers.append((lo + hi) / 2.0)
        bin_means.append(values[mask].mean())

    if not bin_centers:
        return np.array([]), np.array([])

    return np.array(bin_centers), np.array(bin_means)


def plot_learning_curves(
    df_all: pd.DataFrame,
    bin_size: int,
    run_dir: Path,
    save_figs: bool,
):
    """
    Create and optionally save:
      1) Episode length vs timesteps (binned)
      2) Episode return vs timesteps (binned)

    Uses:
      - timesteps = cumulative sum of 'l' (episode lengths) over sorted episodes
      - values = 'l' for episode length curve, 'r' for return curve
    """
    # Build global timesteps as cumulative sum of episode lengths
    ep_len = df_all["l"].to_numpy(dtype=float)
    ep_ret = df_all["r"].to_numpy(dtype=float)

    timesteps = np.cumsum(ep_len)

    # Bin curves
    x_len, y_len = bin_by_timesteps(timesteps, ep_len, bin_size)
    x_ret, y_ret = bin_by_timesteps(timesteps, ep_ret, bin_size)

    # Noise plot
    max_time = timesteps[-1]
    max_noise = 0.014
    x_noise = np.array([0, 0.75*max_time, x_len[-1]])
    y_noise = np.array([0, max_noise, max_noise])

    if len(x_len) == 0 or len(x_ret) == 0:
        print("Warning: Not enough data to produce binned curves.")
    
    # --- Plot episode length ---
    fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Episode length curve
    ax[0].plot(x_len, y_len, label="RL: Average Episode Length (binned)")
    ax[0].axhline(
        y=7.6,
        linestyle="--",
        label="No RL: Average Episode Length",
        color="red"
    )
    ax[0].set_ylabel("Episode length (steps)")
    ax[0].set_ylim(5, 16)

    ax2 = ax[0].twinx()
    ax2.plot(
        x_noise,
        y_noise,
        linestyle="--",
        color="tab:orange",      # not blue or red
        label="Training Noise Standard Deviation",
    )

    ax2.set_ylabel("Noise std (m)")
    ax2.set_ylim(0.0, 0.022)
    ax2.yaxis.set_major_locator(plt.MaxNLocator(7))
    lines1, labels1 = ax[0].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    ax[0].legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    ax[0].grid(True, alpha=0.3)
    ax[0].set_title("Episode Length Over Training Alongside Baseline and Noise Schedule",
                    fontsize=14,
                    y=1.02)

    # --- Plot episode return ---
    ax[1].plot(x_ret, y_ret, label="RL: Average Episode Reward")
    ax[1].set_xlabel("Timesteps")
    ax[1].set_ylabel("Reward")
    ax[1].grid(True, alpha=0.3)
    ax[1].legend()

    ax[1].set_title("Rewards Over Training",
                    fontsize=14,
                    y=1.02)

    fig.suptitle("Learning Curves",
                fontsize=22,
                fontweight="bold",
                y=0.94)
    
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_figs:
        out_path = run_dir / "learning_curves.png"
        fig.savefig(out_path, dpi=200)
        print(f"Saved figure to {out_path}")
    else:
        plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="Plot learning curves from SB3 Monitor logs")
    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Path to a run directory, e.g. ./exp/ppo_foot_residual/seed_015",
    )
    parser.add_argument(
        "--bin_size",
        type=int,
        default=10_000,
        help="Bin size in timesteps for smoothing",
    )
    parser.add_argument(
        "--save_figs",
        action="store_true",
        help="Save figures to disk instead of showing them",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()

    # We expect monitor files under run_dir/train_logs/**.monitor.csv
    train_logs_dir = run_dir / "train_logs"
    if not train_logs_dir.exists():
        raise FileNotFoundError(f"train_logs directory not found under {run_dir}")

    df_all = load_all_monitors(train_logs_dir)
    print(f"Loaded {len(df_all)} episodes from monitor files under {train_logs_dir}")

    plot_learning_curves(
        df_all=df_all,
        bin_size=args.bin_size,
        run_dir=run_dir,
        save_figs=args.save_figs,
    )


if __name__ == "__main__":
    main()
