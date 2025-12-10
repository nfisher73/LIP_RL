"""
LIP model and gait parameter definitions (Kajita 2001 style).

This module defines:
- LIPParams: a dataclass holding physical and gait parameters
- create_default_lip_params: a helper to construct and optionally override them

The sagittal (x) parameters are derived so that, for a given
step_length and timing (T_ss, T_ds), the 1-D LIPM has a
steady-state gait: the initial COM state at each step and the
terminal desired state are constant from step to step.
"""

from dataclasses import dataclass, field
from typing import Any
import numpy as np


@dataclass
class LIPParams:
    """
    Parameters for LIP / 3D-LIPM gait generation.

    Notation follows Kajita et al. 2001:
    """

    # Physical params
    g: float = 9.81
    z_c: float = 0.8
    m: float = 100

    # Derived (do not set manually)
    omega: float = field(init=False)
    T_c: float = field(init=False)

    T_ss: float = 0.7   # single support duration

    num_steps: int = 15
    
    
    s_x: float = 0.3
    s_y: float = 0.2

    x0_rel: float = 0
    vx0: float = 0 

    y0_rel: float = 0.01  
    vy0: float = 0.0                


    # Error norm weights
    a_weight: float = 10.0
    b_weight: float = 1.0

    # Sampling
    dt: float = 0.01

    L_max: float = 0.5

    Q: np.ndarray = field(
        default_factory=lambda: np.diag([10.0, 1.0, 10.0, 1.0])
    )
    R: np.ndarray = field(
        default_factory=lambda: np.diag([0.1, 0.1])
    )

    def __post_init__(self) -> None:
        """
        Compute derived parameters (T_c, omega) and set a consistent
        steady-state sagittal gait (x0_rel, vx0, x_d_rel, v_des_x),
        plus symmetric lateral layout.
        """
        # Time constant and natural frequency
        self.T_c = float(np.sqrt(self.z_c / self.g))
        self.omega = float(np.sqrt(self.g / self.z_c))

       


def create_default_lip_params(**overrides: Any) -> LIPParams:
    """
    Create a LIPParams instance with reasonable defaults, optionally
    overriding any fields.

    Example Usage:
    params = create_default_lip_params(
        T_ss=0.8,
        T_ds=0.12,
    )
    """
    params = LIPParams()

    # Apply user overrides
    for name, value in overrides.items():
        if not hasattr(params, name):
            raise AttributeError(f"Unknown LIPParams field '{name}'")
        setattr(params, name, value)

    # Recompute all derived quantities (T_c, omega, sagittal/lateral states)
    params.__post_init__()

    return params
