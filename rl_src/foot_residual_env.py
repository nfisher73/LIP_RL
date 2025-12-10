from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from include import (
    LIPParams,
    GaitPlan,
    Footstep,
    CoMSegment,
    create_default_lip_params,
    discretize_sys_zoh,
    solve_dare,
)

from analytic_src.straight_walk_ref import generate_straight_gait
from .lip_step_simulator import simulate_one_step
from .rewards import compute_step_reward, RewardWeights


@dataclass
class EnvConfig:
    """
    Small env-level config
    """
    max_foot_residual_x: float = 0.04*100
    max_foot_residual_y: float = 0.04*100
    max_steps_per_episode: int | None = None

    obs_pos_bound: float = 2.0
    obs_vel_bound: float = 2
    foot_noise_std: float = 0.014
    obs_acc_bound: float = 10.0

    curriculum_total_steps: int | None = None#750000
    min_noise_frac: float = 0.002


class LIPFootResidualEnv(gym.Env):
    """
    Gymnasium-compatible environment for LIP + LQR walking with
    residual corrections on foot placements.

    - Action: residual on NEXT foot placement (Δx, Δy) in stance coordinates.
      Since all feet are yaw=0, stance coordinates == world coordinates.

    - On each step():
        1) Take residual action a = [Δx, Δy].
        2) Apply it to the NEXT reference foot placement to obtain the
           true next foot position.
        3) Simulate ONE single-support phase of the LIP + DLQR system
           using `simulate_one_step`.
        4) Compute reward via `compute_step_reward`.
        5) Check fall (COM-to-stance distance > L_max) and episode termination.

    This file is purely the RL environment wrapper. The actual
    single-step simulation, reward shaping, and normalization/details live in:
        - rl_src.lip_step_simulator
        - rl_src.rewards
        - rl_src.utils
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(
        self,
        params: LIPParams = None,
        env_config: EnvConfig = None,
        dist_fn = None,
        render_mode = None,
    ) -> None:
        super().__init__()

        self.params: LIPParams = params if params is not None else create_default_lip_params()
        self.env_config = env_config if env_config is not None else EnvConfig()
        if self.env_config.max_steps_per_episode is None:
            self.env_config.max_steps_per_episode = self.params.num_steps

        self.render_mode = render_mode
        self.dist_fn = dist_fn  # optional disturbance on virtual inputs

        # Internal state
        self.ref_plan = None
        self.true_footsteps = []
        self.com_state = None  
        self.current_segment_idx = 0
        self._terminated = False
        self._truncated = False

        self.rng = np.random.default_rng()

        self.action_space = self._build_action_space()
        self.observation_space = self._build_observation_space()
        self.last_mean_pos_error = np.zeros(2, dtype=np.float32)
        self.rew_weights = RewardWeights()

        self.prev_com_state = None
        self.prev_action = np.zeros(2, dtype=np.float32)

        self.A_d, self.B_d = discretize_sys_zoh(self.params)
        _, self.K = solve_dare(self.A_d, self.B_d, self.params.Q, self.params.R)

        self._global_step = 0
        self.ema_pos_error = 0.0
        self.ema_vel_error = 0.0



    def _build_action_space(self) -> spaces.Box:
        """
        Action = [Δx, Δy] foot placement residual in stance/world frame.
        """
        cfg = self.env_config
        low = np.array(
            [-cfg.max_foot_residual_x, -cfg.max_foot_residual_y],
            dtype=np.float32,
        )
        high = np.array(
            [cfg.max_foot_residual_x, cfg.max_foot_residual_y],
            dtype=np.float32,
        )
        return spaces.Box(low=low, high=high, dtype=np.float32)



    def _build_observation_space(self) -> spaces.Box:
        """
        obs = [
            x_rel_belief, xdot, y_rel_belief, ydot,
            x_rel_ref0,  xdot_ref0, y_rel_ref0, ydot_ref0,
            phase_progress,
            mean_ex, mean_ey,
            dx_step_ref, dy_step_ref,      # step length (x) and width (y)
            dx_next, dy_next,              # COM-to-next-ref-foot
            x_acc, y_acc,                  # NEW: COM acceleration
            prev_dx_res, prev_dy_res,      # NEW: previous action
        ]
        """
        cfg = self.env_config
        pos_bound = cfg.obs_pos_bound
        vel_bound = cfg.obs_vel_bound
        acc_bound = cfg.obs_acc_bound

        lows = np.array(
            [
                -pos_bound, -vel_bound, -pos_bound, -vel_bound,  # belief
                -pos_bound, -vel_bound, -pos_bound, -vel_bound,  # ref at seg start
                -pos_bound, -pos_bound,                          # mean_ex, mean_ey
                -pos_bound, -pos_bound,                          # dx_step_ref, dy_step_ref
                -pos_bound, -pos_bound,                          # dx_next, dy_next
                -1,
                -pos_bound, -pos_bound,
                -1,
                # -acc_bound, -acc_bound,                          # x_acc, y_acc
                # -cfg.max_foot_residual_x,                        # prev_dx_res
                # -cfg.max_foot_residual_y,                        # prev_dy_res
            ],
            dtype=np.float32,
        )

        highs = np.array(
            [
                pos_bound, vel_bound, pos_bound, vel_bound,
                pos_bound, vel_bound, pos_bound, vel_bound,
                pos_bound, pos_bound,
                pos_bound, pos_bound,
                pos_bound, pos_bound,
                1,
                pos_bound, pos_bound,
                1,
                # acc_bound, acc_bound,
                # cfg.max_foot_residual_x,
                # cfg.max_foot_residual_y,
            ],
            dtype=np.float32,
        )

        return spaces.Box(low=lows, high=highs, dtype=np.float32)




    def reset(self, *, seed = None, options = None,) -> tuple:

        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.ref_plan = generate_straight_gait(self.params)

        # Copy over just as starting steps
        self.true_footsteps = [
            Footstep(
                x=f.x,
                y=f.y,
                yaw=f.yaw,
                length=f.length,
                width=f.width,
                is_left=f.is_left,
            )
            for f in self.ref_plan.footsteps
        ]

        # Initialize CoM
        first_seg = self.ref_plan.segments[0]
        x0 = first_seg.x[0]
        xdot0 = first_seg.xdot[0]
        y0 = first_seg.y[0]
        ydot0 = first_seg.ydot[0]

        self.com_state = np.array([x0, xdot0, y0, ydot0], dtype=np.float32)
        self.current_segment_idx = 0
        self._terminated = False
        self._truncated = False
        self.last_mean_pos_error = np.zeros(2, dtype=np.float32)
        self.prev_com_state = None
        self.prev_action = np.zeros(2, dtype=np.float32)
        self.ema_pos_error = 0.0
        self.ema_vel_error = 0.0



        obs = self._get_obs()
        info = {
            "segment_idx": self.current_segment_idx,
            "ref_plan": self.ref_plan,
        }
        return obs, info



    def step(
        self, action: np.ndarray
    ) -> tuple:
        """
        Apply a foot placement residual to next foot step
        """
        if self._terminated or self._truncated:
            raise RuntimeError(
                "step() called on terminated/truncated episode. "
                "Call reset() before stepping again."
            )

        assert self.ref_plan is not None
        assert self.com_state is not None

        prev_com_state = self.com_state.copy()

        # Clip & cast action
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (2,):
            raise ValueError(f"Action must be shape 2, got {action.shape}")
        action = np.clip(action, self.action_space.low, self.action_space.high)
        action /= 100

        seg_idx = self.current_segment_idx
        max_seg_idx = len(self.ref_plan.segments) - 1
        if seg_idx > max_seg_idx:
            self._truncated = True
            obs = self._get_obs()
            return obs, 0.0, self._terminated, self._truncated, {"reason": "no_more_segments"}

        ref_seg = self.ref_plan.segments[seg_idx]
        stance_idx = ref_seg.stance_foot_index

        stance_foot_ref = self.ref_plan.footsteps[stance_idx]
        stance_foot_true = self.true_footsteps[stance_idx]

        # Next stance foot index. If at last foot, clamp.
        if stance_idx + 1 >= len(self.true_footsteps):
            self._truncated = True
            obs = self._get_obs()
            return obs, 0.0, self._terminated, self._truncated, {"reason": "no_next_foot"}
        

        next_foot_idx = stance_idx + 1
        next_foot_ref = self.ref_plan.footsteps[next_foot_idx]

        dx_ref = next_foot_ref.x - stance_foot_ref.x
        dy_ref = next_foot_ref.y - stance_foot_ref.y
        dx_res, dy_res = action[0], action[1]

        self._global_step += 1
        cfg = self.env_config

        if cfg.curriculum_total_steps is not None and cfg.curriculum_total_steps > 0:
            progress = min(1.0, self._global_step / cfg.curriculum_total_steps)
        else:
            progress = 1.0

        noise_frac = cfg.min_noise_frac + (1.0 - cfg.min_noise_frac) * progress
        foot_noise_std = noise_frac * cfg.foot_noise_std

        dx_noise = self.rng.normal(0.0, foot_noise_std)
        dy_noise = self.rng.normal(0.0, foot_noise_std)

        next_foot_true = Footstep(
            x=stance_foot_true.x + dx_ref + dx_res + dx_noise,
            y=stance_foot_true.y + dy_ref + dy_res + dy_noise,
            yaw=next_foot_ref.yaw,
            length=next_foot_ref.length,
            width=next_foot_ref.width,
            is_left=next_foot_ref.is_left,
        )
        # Store/update this in true_footsteps so future segments use it
        self.true_footsteps[next_foot_idx] = next_foot_true
        foot_err_x = next_foot_true.x - next_foot_ref.x
        foot_err_y = next_foot_true.y - next_foot_ref.y
        foot_error = float(np.sqrt(foot_err_x**2 + foot_err_y**2))

        # Simulate one step
        step_result = simulate_one_step(
            com_state=self.com_state,
            ref_segment=ref_seg,
            stance_foot_true=stance_foot_true,
            stance_foot_ref=stance_foot_ref,
            params=self.params,
            A_d = self.A_d,
            B_d = self.B_d,
            K = self.K,
            dist_fn=self.dist_fn,
        )

        next_com_state = step_result["next_com_state"]
        fell = step_result.get("fell", False)
        max_leg_length = step_result.get("max_leg_length", 0.0)
        pos_error = step_result.get("pos_error", 0.0)
        vel_error = step_result.get("vel_error", 0.0)
        mean_pos_error = step_result.get("mean_pos_error", None)
        if mean_pos_error is not None:
            self.last_mean_pos_error = mean_pos_error

        alpha_pos = 0.9 
        alpha_vel = 0.9
        self.ema_pos_error = alpha_pos * self.ema_pos_error + (1.0 - alpha_pos) * pos_error
        self.ema_vel_error = alpha_vel * self.ema_vel_error + (1.0 - alpha_vel) * vel_error


        self.prev_com_state = prev_com_state
        self.com_state = next_com_state.astype(np.float32)
        self.prev_action = action.copy()
        self.current_segment_idx += 1

        # Episode termination / truncation
        self._terminated = fell
        self._truncated = (
            self.current_segment_idx >= self.env_config.max_steps_per_episode
        )
        cfg = self.env_config
        max_norm = np.sqrt(cfg.max_foot_residual_x**2 + cfg.max_foot_residual_y**2)


        # Compute reward from this step
        reward = compute_step_reward(
            pos_error=self.ema_pos_error,
            vel_error = self.ema_vel_error,
            action=action,
            fell=fell,
            max_leg_length=max_leg_length,
            foot_error = foot_error,
            params=self.params,
            weights = self.rew_weights,
            max_action_norm= max_norm,
            current_std=foot_noise_std
        )

        obs = self._get_obs()
        info = {
            "fell": fell,
            "max_leg_length": max_leg_length,
            "pos_error": self.ema_pos_error,
            "vel_error": self.ema_pos_error,
            "segment_idx": self.current_segment_idx,
            "mean_pos_error": self.last_mean_pos_error,
        }

        return obs, float(reward), self._terminated, self._truncated, info


    def _get_obs(self) -> np.ndarray:
        assert self.ref_plan is not None
        assert self.com_state is not None

        seg_idx = min(self.current_segment_idx, len(self.ref_plan.segments) - 1)
        ref_seg = self.ref_plan.segments[seg_idx]
        stance_idx = ref_seg.stance_foot_index
        stance_foot_ref = self.ref_plan.footsteps[stance_idx]

        stance_sign = 1.0 if stance_foot_ref.is_left else -1.0

        x, xdot, y, ydot = self.com_state

        # Belief stance-relative COM state
        x_rel_belief = x - stance_foot_ref.x
        y_rel_belief = y - stance_foot_ref.y

        # Reference COM at start of this segment
        x_ref0 = ref_seg.x[0]
        xdot_ref0 = ref_seg.xdot[0]
        y_ref0 = ref_seg.y[0]
        ydot_ref0 = ref_seg.ydot[0]

        x_rel_ref0 = x_ref0 - stance_foot_ref.x
        y_rel_ref0 = y_ref0 - stance_foot_ref.y

        # Next reference foot
        next_idx = min(stance_idx + 1, len(self.ref_plan.footsteps) - 1)
        next_foot_ref = self.ref_plan.footsteps[next_idx]

        # Step length (x) and width (y) in reference frame
        dx_step_ref = next_foot_ref.x - stance_foot_ref.x
        dy_step_ref = next_foot_ref.y - stance_foot_ref.y

        # COM-to-next-ref-foot vector
        dx_next = next_foot_ref.x - x
        dy_next = next_foot_ref.y - y

        # Mean tracking error from last step
        mean_ex, mean_ey = self.last_mean_pos_error

        omega = self.params.omega
        xi_x_rel = x_rel_belief + xdot / omega
        xi_y_rel = y_rel_belief + ydot / omega

        max_steps = max(1, self.env_config.max_steps_per_episode)
        step_idx_norm = float(self.current_segment_idx) / float(max_steps)
        step_idx_norm = np.clip(step_idx_norm, 0.0, 1.0)

        # COM acceleration (finite diff in velocity)
        # if self.prev_com_state is not None:
        #     _, xdot_prev, _, ydot_prev = self.prev_com_state
        #     dt = self.params.dt
        #     x_acc = (xdot - xdot_prev) / dt
        #     y_acc = (ydot - ydot_prev) / dt
        # else:
        #     x_acc = 0.0
        #     y_acc = 0.0

        # Previous action (residuals)
        # prev_dx_res, prev_dy_res = self.prev_action

        obs = np.array(
            [
                x_rel_belief,
                xdot,
                y_rel_belief,
                ydot,
                x_rel_ref0,
                xdot_ref0,
                y_rel_ref0,
                ydot_ref0,
                mean_ex,
                mean_ey,
                dx_step_ref,
                dy_step_ref,
                dx_next,
                dy_next,
                stance_sign,
                xi_x_rel,
                xi_y_rel,
                step_idx_norm,
                # x_acc,
                # y_acc,
                # prev_dx_res,
                # prev_dy_res,
            ],
            dtype=np.float32,
        )

        return obs
