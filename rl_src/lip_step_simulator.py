import numpy as np
from include.gait_classes import CoMSegment, Footstep
from include.params import LIPParams
# from include.lqr import discretize_sys_zoh, solve_dare
from include.dynamics import compute_virtual_inp_lim


def simulate_one_step(
    com_state,           # np.array([x, xdot, y, ydot])
    ref_segment: CoMSegment,
    stance_foot_true: Footstep,
    stance_foot_ref: Footstep,
    params: LIPParams,
    A_d: np.ndarray,
    B_d: np.ndarray,
    K: np.ndarray,
    dist_fn=None,
    cl: bool = True
) -> dict:
    # loop over k in segment
    # compute belief state (using ref foot)
    # LQR control on belief
    # clip with compute_virtual_inp_lim
    # add disturbance
    # propagate true state (using true foot)
    # track max_leg_length, tracking_error, fall

    # A_d, B_d = discretize_sys_zoh(params)
    # Q = params.Q 
    # R = params.R 
    # P, K = solve_dare(A_d, B_d, Q, R)

    x, xdot, y, ydot = com_state
    p_ref_x = stance_foot_ref.x
    p_ref_y = stance_foot_ref.y
    p_true_x = stance_foot_true.x
    p_true_y = stance_foot_true.y

    x_ref = ref_segment.x
    xdot_ref = ref_segment.xdot
    y_ref = ref_segment.y
    ydot_ref = ref_segment.ydot

    max_dist = 0
    L_max = params.L_max
    fell = False
    pos_err_sq = 0.0
    vel_err_sq = 0.0
    N = len(ref_segment.t)
    sum_x_err = 0.0
    sum_y_err = 0.0

    for i in range(N):
        s_belief = np.array([x - p_ref_x, xdot, y - p_ref_y, ydot])
        s_ref = np.array([x_ref[i] - p_ref_x, xdot_ref[i], y_ref[i] - p_ref_y, ydot_ref[i]])
        s_true = np.array([x - p_true_x, xdot, y - p_true_y, ydot])

        e = s_belief - s_ref

        e_x = x - x_ref[i]
        e_y = y - y_ref[i]
        e_vx = xdot - xdot_ref[i]
        e_vy = ydot - ydot_ref[i]

        pos_err_sq += e_x**2 + e_y**2
        vel_err_sq += e_vx**2 + e_vy**2
        sum_x_err += e_x 
        sum_y_err += e_y

        if cl:
            u = -K @ e
        else:
            u = np.zeros(2)

        x_rel_true = s_true[0]
        y_rel_true = s_true[2]
        u_p_max, u_r_max = compute_virtual_inp_lim(x_rel_true, y_rel_true, params)

        u_p = np.clip(u[0], -u_p_max, u_p_max)
        u_r = np.clip(u[1], -u_r_max, u_r_max)
        u = np.array([u_p, u_r])


        if dist_fn is not None:
            u += dist_fn()

        s_next = A_d @ s_true + B_d @ u
        x_rel_next, xdot, y_rel_next, ydot = s_next
        x = x_rel_next + p_true_x
        y = y_rel_next + p_true_y

        dist = np.sqrt(x_rel_next**2 + y_rel_next**2)
        max_dist = max(max_dist, dist)

        if dist > L_max and not fell:
            fell = True
            break

    steps_used = i + 1 if fell else N
    pos_err = np.sqrt(pos_err_sq/steps_used)
    vel_err = np.sqrt(vel_err_sq/steps_used)
    mean_ex = sum_x_err/steps_used 
    mean_ey = sum_y_err/steps_used
    mean_pos_err = np.array([mean_ex, mean_ey])

    next_com_state = np.array([x, xdot, y, ydot])


    return {
        "next_com_state": next_com_state, 
        "fell": fell,
        "max_leg_length": max_dist,
        "pos_error": pos_err,
        "vel_error": vel_err,
        "mean_pos_error": mean_pos_err
    }