from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from include.params import LIPParams

@dataclass
class RewardWeights:
    '''
    Hyper-parameters for reward function
    '''
    fall_penalty: float = 10
    alive_bonus: float = 4

    w_foot: float = 10.0
    w_pos: float = 0#.4
    w_vel: float = 0#.5
    w_leg: float = 0#1
    w_act: float = 0#.5



def compute_step_reward(
    *,
    pos_error: float,
    vel_error: float,
    action: np.ndarray,
    fell: bool,
    max_leg_length: float,
    foot_error: float,
    params: LIPParams,
    weights: RewardWeights = None,
    max_action_norm=0.056,
    current_std = 0.014
) -> float:
    """
    Returns reward value
    """
    if weights is None:
        weights = RewardWeights()

    reward = 0.0


    if fell:
        reward -= weights.fall_penalty
    else:
        reward += weights.alive_bonus

    leg_utilization = max_leg_length / params.L_max


    action_scaled = float((np.linalg.norm(action) / max_action_norm)**2)
    foot_err_scaled = float((foot_error - 2*current_std) / 0.04)
    
    pos_err_scaled = pos_error / params.s_x

    reward -= weights.w_pos * float(pos_err_scaled)
    reward -= weights.w_vel * float(vel_error)
    reward -= weights.w_leg * (leg_utilization ** 2)
    reward -= weights.w_act * action_scaled
    reward -= weights.w_foot * (foot_err_scaled ** 2)

    return float(reward)
