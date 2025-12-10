from __future__ import annotations

from include.params import LIPParams
from include.dynamics import integrate_lip
from include.primitives import (
    calc_next_step,
    calc_walk_primitive,
    calc_target_state,
    calc_modified_foot_placement,
)
from include.gait_classes import GaitPlan, CoMSegment, Footstep
import numpy as np

def generate_straight_gait(params: LIPParams) -> GaitPlan:

    x = params.x0_rel
    y = params.y0_rel
    T_ss = params.T_ss
    s_x = params.s_x
    s_y = params.s_y

    initial_foot = Footstep(0, 0, is_left = False)

    sx_0 = 0
    sy_0 = 0.2

    n = 0
    T = 0

    x_t, xdot_t, times = integrate_lip(x, 0, 0, params)
    y_t, ydot_t, __ = integrate_lip(y, 0, 0, params)

    com_segs = [CoMSegment(phase = "ss", t = times, x = x_t, y = y_t, xdot = xdot_t, ydot = ydot_t, step_index = n, stance_foot_index = n)]
    footsteps = [initial_foot]

    T += T_ss
    n += 1

    ## Calc next foot place
    px, py = calc_next_step(0, 0, sx_0, sy_0, n)
    pos_bar, vel_bar = calc_walk_primitive(s_x, s_y, params, n)
    xbar, ybar = pos_bar
    velx_bar, vely_bar = vel_bar

    x_d, xdot_d = calc_target_state(px, xbar, velx_bar)
    y_d, ydot_d = calc_target_state(py, ybar, vely_bar)

    px_star = calc_modified_foot_placement(x_t[-1], xdot_t[-1], x_d, xdot_d, params)
    py_star = calc_modified_foot_placement(y_t[-1], ydot_t[-1], y_d, ydot_d, params)

    last_foot = Footstep(px_star, py_star)
    footsteps.append(last_foot)

    for i in range(params.num_steps):
        x_t, xdot_t, times = integrate_lip(x_t[-1], xdot_t[-1], px_star, params)
        y_t, ydot_t, _ = integrate_lip(y_t[-1], ydot_t[-1], py_star, params)
        com_segs.append(CoMSegment(phase = "ss", t = times + T, x = x_t, y = y_t, xdot = xdot_t, ydot = ydot_t, step_index = n, stance_foot_index = n))

        T += T_ss
        n += 1

        px, py = calc_next_step(px, py, s_x, s_y, n)
        pos_bar, vel_bar = calc_walk_primitive(s_x, s_y, params, n)
        xbar, ybar = pos_bar
        velx_bar, vely_bar = vel_bar

        x_d, xdot_d = calc_target_state(px, xbar, velx_bar)
        y_d, ydot_d = calc_target_state(py, ybar, vely_bar)

        px_star = calc_modified_foot_placement(x_t[-1], xdot_t[-1], x_d, xdot_d, params)
        py_star = calc_modified_foot_placement(y_t[-1], ydot_t[-1], y_d, ydot_d, params)

        last_foot = Footstep(px_star, py_star, is_left = not last_foot.is_left)
        footsteps.append(last_foot)

    gait_plan = GaitPlan(footsteps = footsteps, segments = com_segs)

    return gait_plan


def generate_noisy_straight_gait(params: LIPParams) -> GaitPlan:

    x = params.x0_rel
    y = params.y0_rel
    T_ss = params.T_ss
    s_x = params.s_x
    s_y = params.s_y
    scale = 0.02

    initial_foot = Footstep(0, 0, is_left = False)

    sx_0 = 0
    sy_0 = 0.2

    n = 0
    T = 0


    x_t, xdot_t, times = integrate_lip(x, 0, 0, params)
    y_t, ydot_t, __ = integrate_lip(y, 0, 0, params)

    com_segs = [CoMSegment(phase = "ss", t = times, x = x_t, y = y_t, xdot = xdot_t, ydot = ydot_t, step_index = n, stance_foot_index = n)]
    footsteps = [initial_foot]

    T += T_ss
    n += 1

    ## Calc next foot place
    px, py = calc_next_step(0, 0, sx_0, sy_0, n)
    pos_bar, vel_bar = calc_walk_primitive(s_x, s_y, params, n)
    xbar, ybar = pos_bar
    velx_bar, vely_bar = vel_bar

    x_d, xdot_d = calc_target_state(px, xbar, velx_bar)
    y_d, ydot_d = calc_target_state(py, ybar, vely_bar)

    x_noisy = x_t[-1] + np.random.normal(loc = 0, scale = scale)
    xdot_noisy = xdot_t[-1] + np.random.normal(loc = 0, scale = scale)
    y_noisy = y_t[-1] + np.random.normal(loc = 0, scale = scale)
    ydot_noisy = ydot_t[-1] + np.random.normal(loc = 0, scale = scale)


    px_star = calc_modified_foot_placement(x_noisy, xdot_noisy, x_d, xdot_d, params)
    py_star = calc_modified_foot_placement(y_noisy, ydot_noisy, y_d, ydot_d, params)

    last_foot = Footstep(px_star, py_star)
    footsteps.append(last_foot)

    for i in range(params.num_steps):
        x_t, xdot_t, times = integrate_lip(x_t[-1], xdot_t[-1], px_star, params)
        y_t, ydot_t, _ = integrate_lip(y_t[-1], ydot_t[-1], py_star, params)
        com_segs.append(CoMSegment(phase = "ss", t = times + T, x = x_t, y = y_t, xdot = xdot_t, ydot = ydot_t, step_index = n, stance_foot_index = n))

        T += T_ss
        n += 1

        px, py = calc_next_step(px, py, s_x, s_y, n)
        pos_bar, vel_bar = calc_walk_primitive(s_x, s_y, params, n)
        xbar, ybar = pos_bar
        velx_bar, vely_bar = vel_bar

        x_d, xdot_d = calc_target_state(px, xbar, velx_bar)
        y_d, ydot_d = calc_target_state(py, ybar, vely_bar)

        x_noisy = x_t[-1] + np.random.normal(loc = 0, scale = scale)
        xdot_noisy = xdot_t[-1] + np.random.normal(loc = 0, scale = scale)
        y_noisy = y_t[-1] + np.random.normal(loc = 0, scale = scale)
        ydot_noisy = ydot_t[-1] + np.random.normal(loc = 0, scale = scale)


        px_star = calc_modified_foot_placement(x_noisy, xdot_noisy, x_d, xdot_d, params) 
        py_star = calc_modified_foot_placement(y_noisy, ydot_noisy, y_d, ydot_d, params)

        last_foot = Footstep(px_star, py_star, is_left = not last_foot.is_left)
        footsteps.append(last_foot)

    gait_plan = GaitPlan(footsteps = footsteps, segments = com_segs)

    return gait_plan



def noisy_feet_gait(params: LIPParams) -> GaitPlan:

    x = params.x0_rel
    y = params.y0_rel
    T_ss = params.T_ss
    s_x = params.s_x
    s_y = params.s_y
    scale = 0.01

    initial_foot = Footstep(0, 0, is_left = False)

    sx_0 = 0
    sy_0 = 0.2

    n = 0
    T = 0

    x_t, xdot_t, times = integrate_lip(x, 0, 0, params)
    y_t, ydot_t, __ = integrate_lip(y, 0, 0, params)

    com_segs = [CoMSegment(phase = "ss", t = times, x = x_t, y = y_t, xdot = xdot_t, ydot = ydot_t, step_index = n, stance_foot_index = n)]
    footsteps = [initial_foot]

    T += T_ss
    n += 1

    ## Calc next foot place
    px, py = calc_next_step(0, 0, sx_0, sy_0, n)
    pos_bar, vel_bar = calc_walk_primitive(s_x, s_y, params, n)
    xbar, ybar = pos_bar
    velx_bar, vely_bar = vel_bar

    x_d, xdot_d = calc_target_state(px, xbar, velx_bar)
    y_d, ydot_d = calc_target_state(py, ybar, vely_bar)

    px_star = calc_modified_foot_placement(x_t[-1], xdot_t[-1], x_d, xdot_d, params) + np.random.normal(loc = 0, scale = scale)
    py_star = calc_modified_foot_placement(y_t[-1], ydot_t[-1], y_d, ydot_d, params) + np.random.normal(loc = 0, scale = scale)

    last_foot = Footstep(px_star, py_star)
    footsteps.append(last_foot)

    for i in range(params.num_steps):
        x_t, xdot_t, times = integrate_lip(x_t[-1], xdot_t[-1], px_star, params)
        y_t, ydot_t, _ = integrate_lip(y_t[-1], ydot_t[-1], py_star, params)
        com_segs.append(CoMSegment(phase = "ss", t = times + T, x = x_t, y = y_t, xdot = xdot_t, ydot = ydot_t, step_index = n, stance_foot_index = n))

        T += T_ss
        n += 1

        px, py = calc_next_step(px, py, s_x, s_y, n)
        pos_bar, vel_bar = calc_walk_primitive(s_x, s_y, params, n)
        xbar, ybar = pos_bar
        velx_bar, vely_bar = vel_bar

        x_d, xdot_d = calc_target_state(px, xbar, velx_bar)
        y_d, ydot_d = calc_target_state(py, ybar, vely_bar)

        px_star = calc_modified_foot_placement(x_t[-1], xdot_t[-1], x_d, xdot_d, params) + np.random.normal(loc = 0, scale = scale)
        py_star = calc_modified_foot_placement(y_t[-1], ydot_t[-1], y_d, ydot_d, params) # + np.random.normal(loc = 0, scale = scale)

        last_foot = Footstep(px_star, py_star, is_left = not last_foot.is_left)
        footsteps.append(last_foot)

    gait_plan = GaitPlan(footsteps = footsteps, segments = com_segs)

    return gait_plan



