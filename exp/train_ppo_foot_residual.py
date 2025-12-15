# exp/train_ppo_foot_residual.py

from __future__ import annotations

import os
import json
import argparse
import random
from pathlib import Path

import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from include.params import create_default_lip_params
from rl_src.foot_residual_env import LIPFootResidualEnv, EnvConfig
from rl_src.rewards import RewardWeights
from torch import nn

def linear_schedule(initial_value: float):
    """
    Linear learning rate (or clip_range) schedule.
    progress_remaining goes from 1.0 to 0.0 during training.
    """
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

def make_env(seed: int, log_dir: str, env_config: EnvConfig):
    """
    Factory that creates one LIPFootResidualEnv wrapped with Monitor,
    logging episode statistics to run_dir/monitor.csv
    """
    def _init():
        params = create_default_lip_params()

        env = LIPFootResidualEnv(params=params, env_config=env_config)

        # IMPORTANT: wrap with Monitor for logging
        os.makedirs(log_dir, exist_ok=True)
        monitor_path = os.path.join(log_dir, f"monitor_seed{seed}")
        env = Monitor(env, filename=monitor_path)

        env.reset(seed=seed)
        return env

    return _init



def save_run_config(run_dir: Path, args, params, env_cfg, rew_weights: RewardWeights):
    """
    Save basic run configuration (hyperparams, env params, reward weights).
    """
    cfg = {
        "seed": args.seed,
        "total_timesteps": args.total_timesteps,
        "learning_rate": args.learning_rate,
        "gamma": args.gamma,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "clip_range": args.clip_range,
        "ent_coef": args.ent_coef,
        "vf_coef": args.vf_coef,
        "max_grad_norm": args.max_grad_norm,
        "walks_per_update": args.walks_per_update,
        "params": {
            "g": params.g,
            "z_c": params.z_c,
            "m": params.m,
            "T_ss": params.T_ss,
            "num_steps": params.num_steps,
            "s_x": params.s_x,
            "s_y": params.s_y,
            "x0_rel": params.x0_rel,
            "vx0": params.vx0,
            "y0_rel": params.y0_rel,
            "vy0": params.vy0,
            "dt": params.dt,
            "L_max": params.L_max,
            "Q": params.Q.tolist(),
            "R": params.R.tolist(),
        },
        "reward_weights": {
            "fall_penalty": rew_weights.fall_penalty,
            "alive_bonus": rew_weights.alive_bonus,
            "w_pos": rew_weights.w_pos,
            "w_vel": rew_weights.w_vel,
            "w_leg": rew_weights.w_leg,
            "w_act": rew_weights.w_act,
        },
        "env_config": {
            "max_foot_residual_x": env_cfg.max_foot_residual_x,
            "max_foot_residual_y": env_cfg.max_foot_residual_y,
            "max_steps_per_episode": env_cfg.max_steps_per_episode,
            "obs_pos_bound": env_cfg.obs_pos_bound,
            "obs_vel_bound": env_cfg.obs_vel_bound,
            "foot_noise_std": env_cfg.foot_noise_std,
            "obs_acc_bound": env_cfg.obs_acc_bound,
            # curriculum params once you add them
        }

    }

    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = run_dir / "config.json"
    with cfg_path.open("w") as f:
        json.dump(cfg, f, indent=2)
    print(f"Saved run config to: {cfg_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO on LIPFootResidualEnv")

    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=1_000_000,
        help="Total PPO training timesteps",
    )
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--walks_per_update", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=96)
    parser.add_argument("--clip_range", type=float, default=0.2)
    parser.add_argument("--ent_coef", type=float, default=0.005)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument(
        "--n_envs",
        type=int,
        default=8,
        help="Number of parallel environments",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # --- Seeding ---
    seed = args.seed
    np.random.seed(seed)
    random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass

    # --- Paths ---
    project_root = Path(__file__).resolve().parents[1]  # go up from exp/
    runs_root = project_root / "exp" / "ppo_foot_residual"
    run_dir = runs_root / f"seed_{seed:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)

    train_log_dir = run_dir / "train_logs"
    eval_log_dir = run_dir / "eval_env_logs"
    train_log_dir.mkdir(parents=True, exist_ok=True)
    eval_log_dir.mkdir(parents=True, exist_ok=True)

    # --- Build envs ---
    # Training env
    n_envs = args.n_envs

    # Grab params & reward weights for logging
    params = create_default_lip_params()
    rew_weights = RewardWeights()

    n_envs = args.n_envs
    total_timesteps = args.total_timesteps
    ramp_fraction = 0.75
    steps_per_env_for_full_noise = int(total_timesteps * ramp_fraction / n_envs)
    env_cfg = EnvConfig(curriculum_total_steps=steps_per_env_for_full_noise)

    walks_per_update = args.walks_per_update
    steps_per_walk = params.num_steps  # episode length in steps (one step per env.step)
    n_steps = walks_per_update * steps_per_walk
    args.n_steps = n_steps

    train_env_fns = [
        make_env(seed + i, str(train_log_dir / f"env_{i:02d}"), env_cfg)
        for i in range(n_envs)
    ]
    train_env = DummyVecEnv(train_env_fns)

    # Separate eval env (different seed)
    eval_env_cfg = EnvConfig(curriculum_total_steps=1)

    eval_env = DummyVecEnv([
        make_env(seed + 1000, str(eval_log_dir / "env_00"), eval_env_cfg)
    ])


    # Save run configuration
    save_run_config(run_dir, args, params, env_cfg, rew_weights)

    # --- Callbacks ---
    # Save checkpoints periodically into run_dir
    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path=str(run_dir),
        name_prefix="checkpoint",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(run_dir / "best_model"),
        log_path=str(run_dir / "eval_logs"),
        eval_freq=20_000,
        deterministic=True,
        render=False,
    )

    policy_kwargs = dict(
        activation_fn=nn.Tanh,
        net_arch=dict(
            pi=[256, 256, 256],
            vf=[256, 256, 256]
        )
    )


    # --- PPO model ---
    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=linear_schedule(args.learning_rate),
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        verbose=1,
        tensorboard_log=str(runs_root / "tb_logs"),
        seed=seed,
        policy_kwargs=policy_kwargs,
        gae_lambda=0.98
    )

    # --- Train ---
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[checkpoint_callback, eval_callback],
    )

    # --- Save final model ---
    final_path = run_dir / "final_model"
    model.save(str(final_path))
    print(f"Saved final model to: {final_path}")


if __name__ == "__main__":
    main()
