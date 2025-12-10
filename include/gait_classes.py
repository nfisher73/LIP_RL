"""
Common data structures for LIP-based gaits.
"""

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class Footstep:
    """
    Represents a single foot placement on the ground.

    x, y : world coordinates of the foot center.
    yaw  : orientation around z (rad). For now we keep feet axis-aligned,
           but yaw is here for future circular gaits.
    length, width : footprint size (for plotting rectangles).
    is_left : True if left foot, False if right foot.
    """

    x: float
    y: float
    yaw: float = 0.0
    length: float = 0.20
    width: float = 0.10
    is_left: bool = True


@dataclass
class CoMSegment:
    """
    One contiguous segment of the CoM trajectory.

    phase : "SS" (single support) or "DS" (double support)
    t     : absolute time stamps [s]
    x, y  : CoM position in world frame
    step_index : which step this segment belongs to
    stance_foot_index : which foot is the stance foot (index into footsteps list)
                        during this segment (for SS); for DS we can set -1.
    """

    phase: str
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    xdot: np.ndarray
    ydot: np.ndarray
    step_index: int
    stance_foot_index: int


@dataclass
class GaitPlan:
    """
    Full LIP-based gait description for plotting.

    footsteps : list of Footstep
    segments  : list of CoMSegment
    """

    footsteps: List[Footstep]
    segments: List[CoMSegment]
