# visualization/plot_gait.py
"""
Plotting utilities for LIP-based gait plans.

Main function:
- plot_gait_2d: plot CoM x-y trajectory along with footsteps.
"""

from typing import Optional, Iterable

import numpy as np
import matplotlib.pyplot as plt

from include.gait_classes import GaitPlan, CoMSegment
from .plot_utils import draw_footstep


def _collect_com_xy(segments: Iterable[CoMSegment]):
    """Concatenate x and y from all segments for global bounds."""
    all_x = np.concatenate([seg.x for seg in segments]) if segments else np.array([])
    all_y = np.concatenate([seg.y for seg in segments]) if segments else np.array([])
    return all_x, all_y

def _collect_com_xydot(segments: Iterable[CoMSegment]):
    """Concatenate x and y from all segments for global bounds."""
    all_xdot = np.concatenate([seg.xdot for seg in segments]) if segments else np.array([])
    all_ydot = np.concatenate([seg.ydot for seg in segments]) if segments else np.array([])
    return all_xdot, all_ydot


def plot_gait_2d(
    plan: GaitPlan,
    ax: Optional[plt.Axes] = None,
    show: bool = True,
    title: str = "LIP-based gait: CoM trajectory and footsteps",
    equal_aspect: bool = False,
) -> plt.Axes:
    """
    Plot the CoM trajectory in the x-y plane along with footsteps.

    Conventions:
    - Single support (SS) segments are solid lines.
    - Double support (DS) segments are dotted lines.
    - Left and right footsteps are colored differently.
    - Feet are drawn as rectangles using their length/width.

    Parameters
    ----------
    plan : GaitPlan
        Gait plan containing footsteps and CoM segments.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, a new figure and axes are created.
    show : bool
        If True, calls plt.show() at the end.
    title : str
        Plot title.
    equal_aspect : bool
        If True, enforce 1:1 aspect (geometry-accurate but can look squashed).
        If False, let matplotlib choose aspect so the plot is visually nicer.
    """
    # Make the figure reasonably wide and tall
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.0, 4.0))

    # --- Plot CoM trajectory, distinguishing SS vs DS ---

    for seg in plan.segments:
        linestyle = "-" 
        linewidth = 2.0 
        label = None
        if seg.step_index == 0:
            label = f"CoM {seg.phase}"

        ax.plot(
            seg.x,
            seg.y,
            linestyle,
            linewidth=linewidth,
            color="C0",
            alpha=0.9 if seg.phase == "SS" else 0.7,
            label=label,
        )

    # --- Plot footsteps as rectangles ---

    left_label_done = False
    right_label_done = False

    for foot in plan.footsteps:
        color = "C1" if foot.is_left else "C2"
        label = None
        if foot.is_left and not left_label_done:
            label = "Left foot"
            left_label_done = True
        elif (not foot.is_left) and not right_label_done:
            label = "Right foot"
            right_label_done = True

        draw_footstep(ax, foot, edgecolor=color, facecolor=color, alpha=0.4)
        if label is not None:
            ax.plot([], [], color=color, label=label)

    # --- Axes formatting ---

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title)

    # Use both CoM and footsteps to determine axis bounds
    all_x, all_y = _collect_com_xy(plan.segments)

    if plan.footsteps:
        foot_x = np.array([f.x for f in plan.footsteps])
        foot_y = np.array([f.y for f in plan.footsteps])
        all_x = np.concatenate([all_x, foot_x]) if all_x.size > 0 else foot_x
        all_y = np.concatenate([all_y, foot_y]) if all_y.size > 0 else foot_y

    if all_x.size > 0 and all_y.size > 0:
        # Add some padding around data
        dx = max(1e-3, all_x.max() - all_x.min())
        dy = max(1e-3, all_y.max() - all_y.min())
        margin_x = 0.15 * dx
        margin_y = 0.4 * dy   # extra vertical space so it's not cramped

        ax.set_xlim(all_x.min() - margin_x, all_x.max() + margin_x)
        ax.set_ylim(all_y.min() - margin_y, all_y.max() + margin_y)

    if equal_aspect:
        ax.set_aspect("equal", adjustable="box")
    else:
        ax.set_aspect("auto")

    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

    # Legend: one combined legend, outside-ish so it doesn't cover data
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    if unique:
        ax.legend(
            unique.values(),
            unique.keys(),
            loc="upper right",
            frameon=True,
            fontsize=9,
        )

    # Tight layout to reduce extra margins
    if ax.figure is not None:
        ax.figure.tight_layout()

    if show:
        plt.show()

    return ax

def plot_2_gaits(
    plan: GaitPlan,
    ax: Optional[plt.Axes] = None,
    show: bool = True,
    title: str = "LIP-based gait: CoM trajectory and footsteps",
    equal_aspect: bool = False,
    extra_segments = None,  # NEW: optional second CoM trajectory
    extra_label: str = "CoM (alt)",                     # NEW: label for second trajectory
    extra_color: str = "C3",                            # NEW: color for second trajectory
) -> plt.Axes:
    """
    Plot the CoM trajectory in the x-y plane along with footsteps.

    Conventions:
    - Single support (SS) segments are solid lines.
    - Double support (DS) segments are dotted lines.
    - Left and right footsteps are colored differently.
    - Feet are drawn as rectangles using their length/width.

    Parameters
    ----------
    plan : GaitPlan
        Gait plan containing footsteps and CoM segments.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, a new figure and axes are created.
    show : bool
        If True, calls plt.show() at the end.
    title : str
        Plot title.
    equal_aspect : bool
        If True, enforce 1:1 aspect (geometry-accurate but can look squashed).
        If False, let matplotlib choose aspect so the plot is visually nicer.
    extra_segments : list[CoMSegment], optional
        If provided, plot this second set of CoM segments on the same axes.
    extra_label : str
        Legend label prefix for the extra CoM trajectory.
    extra_color : str
        Matplotlib color for the extra CoM trajectory.
    """
    # Make the figure reasonably wide and tall
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.0, 4.0))

    # --- Plot CoM trajectory, distinguishing SS vs DS ---

    for seg in plan.segments:
        linestyle = "-"    # could switch on seg.phase if you want
        linewidth = 2.0
        label = None
        if seg.step_index == 0:
            label = f"CoM {seg.phase}"

        ax.plot(
            seg.x,
            seg.y,
            linestyle,
            linewidth=linewidth,
            color="C0",
            alpha=0.9 if seg.phase == "SS" else 0.7,
            label=label,
        )

    # --- NEW: plot extra CoM segments (e.g. reference trajectory) ---

    if extra_segments is not None:
        extra_label_done = False
        for seg in extra_segments:
            linestyle = "--"
            linewidth = 1.8
            label = None
            if not extra_label_done and seg.step_index == 0:
                label = extra_label
                extra_label_done = True

            ax.plot(
                seg.x,
                seg.y,
                linestyle,
                linewidth=linewidth,
                color=extra_color,
                alpha=0.9 if seg.phase == "SS" else 0.7,
                label=label,
            )

    # --- Plot footsteps as rectangles ---

    left_label_done = False
    right_label_done = False

    for foot in plan.footsteps:
        color = "C1" if foot.is_left else "C2"
        label = None
        if foot.is_left and not left_label_done:
            label = "Left foot"
            left_label_done = True
        elif (not foot.is_left) and not right_label_done:
            label = "Right foot"
            right_label_done = True

        draw_footstep(ax, foot, edgecolor=color, facecolor=color, alpha=0.4)
        if label is not None:
            ax.plot([], [], color=color, label=label)

    # --- Axes formatting ---

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(title)

    # Use CoM (both sets) and footsteps to determine axis bounds
    all_x, all_y = _collect_com_xy(plan.segments)

    # NEW: incorporate extra_segments into bounds
    if extra_segments is not None:
        extra_x, extra_y = _collect_com_xy(extra_segments)
        if extra_x.size > 0:
            all_x = np.concatenate([all_x, extra_x]) if all_x.size > 0 else extra_x
            all_y = np.concatenate([all_y, extra_y]) if all_y.size > 0 else extra_y

    if plan.footsteps:
        foot_x = np.array([f.x for f in plan.footsteps])
        foot_y = np.array([f.y for f in plan.footsteps])
        all_x = np.concatenate([all_x, foot_x]) if all_x.size > 0 else foot_x
        all_y = np.concatenate([all_y, foot_y]) if all_y.size > 0 else foot_y

    if all_x.size > 0 and all_y.size > 0:
        # Add some padding around data
        dx = max(1e-3, all_x.max() - all_x.min())
        dy = max(1e-3, all_y.max() - all_y.min())
        margin_x = 0.15 * dx
        margin_y = 0.4 * dy   # extra vertical space so it's not cramped

        ax.set_xlim(all_x.min() - margin_x, all_x.max() + margin_x)
        ax.set_ylim(all_y.min() - margin_y, all_y.max() + margin_y)

    if equal_aspect:
        ax.set_aspect("equal", adjustable="box")
    else:
        ax.set_aspect("auto")

    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.5)

    # Legend: one combined legend, outside-ish so it doesn't cover data
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    if unique:
        ax.legend(
            unique.values(),
            unique.keys(),
            loc="upper right",
            frameon=True,
            fontsize=9,
        )

    # Tight layout to reduce extra margins
    if ax.figure is not None:
        ax.figure.tight_layout()

    if show:
        plt.show()

    return ax


def plot_gait_vel_2d(
    plan: GaitPlan,
    show: bool = True,
    title: str = "CoM Velocities",
    step_period: float = 0.7,   # draw dashed lines every 0.7 seconds
):
    """
    Plot x and y COM velocities over time using two subplots.
    Ignores footsteps. Adds vertical dashed lines every step_period seconds.
    """

    # Create figure with two subplots stacked vertically
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    fig.suptitle(title)

    # --- Collect and concatenate all segments ---
    t_all = []
    xdot_all = []
    ydot_all = []

    for seg in plan.segments:
        t_all.append(seg.t)
        xdot_all.append(seg.xdot)
        ydot_all.append(seg.ydot)

    t_all = np.concatenate(t_all)
    xdot_all = np.concatenate(xdot_all)
    ydot_all = np.concatenate(ydot_all)

    # --- Plot velocities ---
    ax1.plot(t_all, xdot_all, color="C0", linewidth=2)
    ax2.plot(t_all, ydot_all, color="C1", linewidth=2)

    ax1.set_ylabel("x_dot [m/s]")
    ax2.set_ylabel("y_dot [m/s]")
    ax2.set_xlabel("time [s]")

    # --- Add vertical dashed lines every step_period seconds ---
    t_max = t_all[-1]
    print(t_max)
    step_times = np.arange(0.0, t_max, step_period)

    for st in step_times:
        ax1.axvline(st, color="gray", linestyle="--", alpha=0.4)
        ax2.axvline(st, color="gray", linestyle="--", alpha=0.4)

    # --- Grid & formatting ---
    ax1.grid(True, linestyle="--", alpha=0.3)
    ax2.grid(True, linestyle="--", alpha=0.3)

    dx = max(1e-3, t_max - t_all[0])
    dy1 = max(1e-3, xdot_all.max() - xdot_all.min())
    dy2 = max(1e-3, ydot_all.max() - ydot_all.min())
    margin_x = 0.1 * dx
    margin_y1 = 0.4 * dy1   # extra vertical space so it's not cramped
    margin_y2 = 0.4 * dy2

    ax1.set_ylim(xdot_all.min() - margin_y1, xdot_all.max() + margin_y1)
    ax1.set_xlim(t_all[0] - margin_x, t_max + margin_x)
    ax2.set_ylim(ydot_all.min() - margin_y2, xdot_all.max() + margin_y2)
    ax2.set_xlim(t_all[0] - margin_x, t_max + margin_x)


    fig.tight_layout(rect=[0, 0, 1, 0.95])  # leave room for title

    if show:
        plt.show()

    return fig, (ax1, ax2)

