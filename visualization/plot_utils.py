# visualization/plot_utils.py
"""
Helper utilities for plotting LIP-based gaits.

Currently provides:
- draw_footstep: draw a rectangular footprint (optionally rotated).
"""

from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D

from include.gait_classes import Footstep


def draw_footstep(
    ax: plt.Axes,
    foot: Footstep,
    edgecolor: str = "k",
    facecolor: Optional[str] = None,
    alpha: float = 0.4,
    zorder: int = 2,
) -> None:
    """
    Draw a foot rectangle on the given axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to draw on.
    foot : Footstep
        Footstep object defining center (x, y), size, and yaw.
    edgecolor : str
        Color of the rectangle edge.
    facecolor : str or None
        Fill color. If None, uses edgecolor.
    alpha : float
        Transparency for the rectangle face.
    zorder : int
        Z-order for drawing (higher is on top).
    """
    if facecolor is None:
        facecolor = edgecolor

    L = foot.length*0.5
    W = foot.width*0.5

    # Rectangle is defined in local coordinates with lower-left corner.
    # We want the rectangle centered at (foot.x, foot.y), so lower-left is:
    x0 = foot.x - 0.5 * L
    y0 = foot.y - 0.5 * W

    rect = Rectangle(
        (x0, y0),
        L,
        W,
        linewidth=1.0,
        edgecolor=edgecolor,
        facecolor=facecolor,
        alpha=alpha,
        zorder=zorder,
    )

    # If yaw is not zero, apply a rotation about the foot center.
    if abs(foot.yaw) > 1e-9:
        transform = (
            Affine2D()
            .rotate_around(foot.x, foot.y, foot.yaw)
            + ax.transData
        )
        rect.set_transform(transform)

    ax.add_patch(rect)
