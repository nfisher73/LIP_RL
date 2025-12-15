from __future__ import annotations
import numpy as np

from .straight_walk_ref import generate_straight_gait
from include.lqr import discretize_sys_zoh, solve_dare
from include.params import LIPParams
from include.gait_classes import GaitPlan, CoMSegment, Footstep
from include.dynamics import compute_virtual_inp_lim 


def straight_walk(params: LIPParams, cl: bool = True, dist_fn = None, scale = None):
    
    Q = params.Q
    R = params.R

    A_d, B_d = discretize_sys_zoh(params)
    P, K = solve_dare(A_d, B_d, Q, R)

    ref_plan = generate_straight_gait(params)

    x = params.x0_rel
    y = params.y0_rel
    xdot = params.vx0
    ydot = params.vy0

    sim_segments = []

    max_dist = 0
    max_u = 0

    for i, ref_seg in enumerate(ref_plan.segments):
        t_ref = ref_seg.t
        x_ref = ref_seg.x
        y_ref = ref_seg.y
        xdot_ref = ref_seg.xdot
        ydot_ref = ref_seg.ydot

        stance_idx = ref_seg.stance_foot_index
        stance_foot = ref_plan.footsteps[stance_idx]
        p_x = stance_foot.x
        p_y = stance_foot.y

        N = len(t_ref)

        if N > 1:
            dt_seg = t_ref[1] - t_ref[0]
            assert np.allclose(np.diff(t_ref), dt_seg, atol=1e-3)
            assert np.allclose(params.dt, dt_seg, atol=1e-3)

        x_sim = np.zeros(N)
        y_sim = np.zeros(N)
        xdot_sim = np.zeros(N)
        ydot_sim = np.zeros(N)

        for k in range(N):
            x_sim[k] = x
            xdot_sim[k] = xdot
            y_sim[k] = y
            ydot_sim[k] = ydot

            s_rel = np.array([x - p_x, xdot, y - p_y, ydot])
            s_ref_rel = np.array([x_ref[k] - p_x, xdot_ref[k], y_ref[k] - p_y, ydot_ref[k]])
            e = s_rel - s_ref_rel

            if cl:
                u = -K @ e
                if np.max(abs(u)) > max_u:
                    max_u = np.max(abs(u))
            else:
                u = np.zeros(2)

            if dist_fn is not None:
                if scale is not None:
                    u += dist_fn(scale = scale)
                else:
                    u += dist_fn()

            s_rel_next = A_d @ s_rel + B_d @ u
            x_rel, xdot, y_rel, ydot = s_rel_next
            x = x_rel + p_x
            y = y_rel + p_y

            dist = np.sqrt(x_rel**2 + y_rel**2)
            if dist > max_dist:
                max_dist = dist


        sim_seg = CoMSegment(
            phase=ref_seg.phase,
            t=t_ref.copy(),
            x=x_sim,
            xdot=xdot_sim,
            y=y_sim,
            ydot=ydot_sim,
            step_index=ref_seg.step_index,
            stance_foot_index=ref_seg.stance_foot_index,
        )
        sim_segments.append(sim_seg)


    sim_plan = GaitPlan(
        footsteps=ref_plan.footsteps,
        segments=sim_segments,
    )

    print(f'Max Leg Length: {max_dist}\n')
    print(f'Max Input: {max_u}\n')

    return sim_plan


def noisy_straight_walk(params: LIPParams, noise_std: int = 0.1, dist_fn = None):

    Q = params.Q
    R = params.R

    A_d, B_d = discretize_sys_zoh(params)
    P, K = solve_dare(A_d, B_d, Q, R)

    ref_plan = generate_straight_gait(params)

    x = params.x0_rel
    y = params.y0_rel
    xdot = params.vx0
    ydot = params.vy0

    sim_segments = []

    max_dist = 0

    for i, ref_seg in enumerate(ref_plan.segments):
        t_ref = ref_seg.t
        x_ref = ref_seg.x
        y_ref = ref_seg.y
        xdot_ref = ref_seg.xdot
        ydot_ref = ref_seg.ydot

        stance_idx = ref_seg.stance_foot_index
        stance_foot = ref_plan.footsteps[stance_idx]
        p_x = stance_foot.x
        p_y = stance_foot.y

        N = len(t_ref)

        if N > 1:
            dt_seg = t_ref[1] - t_ref[0]
            assert np.allclose(np.diff(t_ref), dt_seg, atol=1e-3)
            assert np.allclose(params.dt, dt_seg, atol=1e-3)

        x_sim = np.zeros(N)
        y_sim = np.zeros(N)
        xdot_sim = np.zeros(N)
        ydot_sim = np.zeros(N)

        for k in range(N):
            x_sim[k] = x
            xdot_sim[k] = xdot
            y_sim[k] = y
            ydot_sim[k] = ydot

            noise = noise_std*np.random.randn(4)

            s_rel = np.array([x - p_x, xdot, y - p_y, ydot])
            s_ref_rel = np.array([x_ref[k] - p_x, xdot_ref[k], y_ref[k] - p_y, ydot_ref[k]])
            e = s_rel - s_ref_rel + noise


            u = -K @ e

            if dist_fn is not None:
                u += dist_fn()

            s_rel_next = A_d @ s_rel + B_d @ u
            x_rel, xdot, y_rel, ydot = s_rel_next
            x = x_rel + p_x
            y = y_rel + p_y

            dist = np.sqrt(x_rel**2 + y_rel**2)
            if dist > max_dist:
                max_dist = dist


        sim_seg = CoMSegment(
            phase=ref_seg.phase,
            t=t_ref.copy(),
            x=x_sim,
            xdot=xdot_sim,
            y=y_sim,
            ydot=ydot_sim,
            step_index=ref_seg.step_index,
            stance_foot_index=ref_seg.stance_foot_index,
        )
        sim_segments.append(sim_seg)


    sim_plan = GaitPlan(
        footsteps=ref_plan.footsteps,
        segments=sim_segments,
    )

    print(f'Max Leg Length: {max_dist}\n')

    return sim_plan



def straight_walk_with_foot_errors(params: LIPParams, cl: bool = True, dist_fn = None, 
                                   foot_noise_std: float = 0.02, prnt: bool = True) -> tuple[GaitPlan, bool, float, int]:
    """
    LQR with foot placement errors
    """

    Q = params.Q
    R = params.R

    A_d, B_d = discretize_sys_zoh(params)
    P, K = solve_dare(A_d, B_d, Q, R)

    ref_plan = generate_straight_gait(params)


    true_footsteps = []

    for i, f_ref in enumerate(ref_plan.footsteps):
        if i == 0:
            # No noise for first step
            true_footsteps.append(
                Footstep(
                    x=f_ref.x,
                    y=f_ref.y,
                    yaw=f_ref.yaw,
                    length=f_ref.length,
                    width=f_ref.width,
                    is_left=f_ref.is_left,
                )
            )

        else:
            f_prev_ref = ref_plan.footsteps[i - 1]
            f_prev_true = true_footsteps[i - 1]

            dx_ref = f_ref.x - f_prev_ref.x
            dy_ref = f_ref.y - f_prev_ref.y

            dx_noise = np.random.normal(0.0, foot_noise_std)
            dy_noise = np.random.normal(0.0, foot_noise_std)

            x_true = f_prev_true.x + dx_ref + dx_noise
            y_true = f_prev_true.y + dy_ref + dy_noise

            true_footsteps.append(
                Footstep(
                    x=x_true,
                    y=y_true,
                    yaw=f_ref.yaw,
                    length=f_ref.length,
                    width=f_ref.width,
                    is_left=f_ref.is_left,
                )
            )

    x = params.x0_rel
    y = params.y0_rel
    xdot = params.vx0
    ydot = params.vy0

    sim_segments = []
    max_dist = 0.0
    max_u = 0.0

    L_max = params.L_max

    fell = False
    fall_time = None
    fall_step = None
    last_used_foot_idx = 0


    for i, ref_seg in enumerate(ref_plan.segments):
        t_ref = ref_seg.t
        x_ref = ref_seg.x
        y_ref = ref_seg.y
        xdot_ref = ref_seg.xdot
        ydot_ref = ref_seg.ydot

        stance_idx = ref_seg.stance_foot_index
        last_used_foot_idx = max(last_used_foot_idx, stance_idx)
        foot_ref = ref_plan.footsteps[stance_idx]
        p_ref_x = foot_ref.x
        p_ref_y = foot_ref.y

        foot_true = true_footsteps[stance_idx]
        p_true_x = foot_true.x
        p_true_y = foot_true.y

        N = len(t_ref)
        if N > 1:
            dt_seg = t_ref[1] - t_ref[0]
            assert np.allclose(np.diff(t_ref), dt_seg, atol=1e-3)

        x_sim = np.zeros(N)
        y_sim = np.zeros(N)
        xdot_sim = np.zeros(N)
        ydot_sim = np.zeros(N)

        for k in range(N):
            x_sim[k] = x
            xdot_sim[k] = xdot
            y_sim[k] = y
            ydot_sim[k] = ydot

            # True stance-relative state
            s_rel_true = np.array([
                x - p_true_x,
                xdot,
                y - p_true_y,
                ydot,
            ])

            # Belief stance-relative state
            s_rel_belief = np.array([
                x - p_ref_x,
                xdot,
                y - p_ref_y,
                ydot,
            ])

            # Reference stance-relative state 
            s_ref_rel = np.array([
                x_ref[k] - p_ref_x,
                xdot_ref[k],
                y_ref[k] - p_ref_y,
                ydot_ref[k],
            ])

            # Error in belief
            e = s_rel_belief - s_ref_rel

            if cl:
                u = -K @ e
            else:
                u = np.zeros(2)

            x_rel_true = s_rel_true[0]
            y_rel_true = s_rel_true[2]
            u_p_max, u_r_max = compute_virtual_inp_lim(x_rel_true, y_rel_true, params)

            u_p = np.clip(u[0], -u_p_max, u_p_max)
            u_r = np.clip(u[1], -u_r_max, u_r_max)
            u = np.array([u_p, u_r])

            max_u = max(max_u, np.max(np.abs(u)))

            # Optional disturbance on virtual inputs
            if dist_fn is not None:
                u = u + dist_fn()

            # Dynamics from true stance frame
            s_rel_next = A_d @ s_rel_true + B_d @ u
            x_rel_next, xdot_next, y_rel_next, ydot_next = s_rel_next

            x = x_rel_next + p_true_x
            y = y_rel_next + p_true_y
            xdot = xdot_next
            ydot = ydot_next

            # Distance from COM to TRUE stance foot
            dist = np.sqrt(x_rel_next**2 + y_rel_next**2)
            max_dist = max(max_dist, dist)

            fall_step = i

            # Fall condition
            if dist > L_max and not fell:
                fell = True
                fall_time = float(t_ref[k])

                # Truncate logs
                x_sim = x_sim[:k + 1]
                y_sim = y_sim[:k + 1]
                xdot_sim = xdot_sim[:k + 1]
                ydot_sim = ydot_sim[:k + 1]
                t_seg = t_ref[:k + 1]

                sim_seg = CoMSegment(
                    phase=ref_seg.phase,
                    t=t_seg.copy(),
                    x=x_sim,
                    xdot=xdot_sim,
                    y=y_sim,
                    ydot=ydot_sim,
                    step_index=ref_seg.step_index,
                    stance_foot_index=ref_seg.stance_foot_index,
                )
                sim_segments.append(sim_seg)
                break 

        if fell:
            break 

        if not fell:
            sim_seg = CoMSegment(
                phase=ref_seg.phase,
                t=t_ref.copy(),
                x=x_sim,
                xdot=xdot_sim,
                y=y_sim,
                ydot=ydot_sim,
                step_index=ref_seg.step_index,
                stance_foot_index=ref_seg.stance_foot_index,
            )
            sim_segments.append(sim_seg)

    used_footsteps = true_footsteps[: last_used_foot_idx + 1]

    sim_plan = GaitPlan(
        footsteps=used_footsteps,
        segments=sim_segments,
    )
    if prnt:
        print(f"Max COM distance to TRUE stance foot: {max_dist:.3f} m")
        print(f"Max virtual input magnitude: {max_u:.3f}")
        if fell:
            print(f"FALL at t = {fall_time:.3f} s, segment index = {fall_step}")
        else:
            print("No fall detected.")



    return sim_plan, fell, fall_time, fall_step



def straight_walk_with_start_foot_error(params: LIPParams, cl: bool = True, dist_fn = None, 
                                   foot_noise_std: float = 0.02, prnt: bool = True) -> tuple[GaitPlan, bool, float, int]:
    """
    LQR with foot placement errors
    """

    Q = params.Q
    R = params.R

    A_d, B_d = discretize_sys_zoh(params)
    P, K = solve_dare(A_d, B_d, Q, R)

    ref_plan = generate_straight_gait(params)


    true_footsteps = []
    dx_noise = foot_noise_std
    dy_noise = foot_noise_std

    for i, f_ref in enumerate(ref_plan.footsteps):
        true_footsteps.append(
            Footstep(
                x=f_ref.x + dx_noise,
                y=f_ref.y + dy_noise,
                yaw=f_ref.yaw,
                length=f_ref.length,
                width=f_ref.width,
                is_left=f_ref.is_left,
            )
        )

    x = params.x0_rel
    y = params.y0_rel
    xdot = params.vx0
    ydot = params.vy0

    sim_segments = []
    max_dist = 0.0
    max_u = 0.0

    L_max = params.L_max

    fell = False
    fall_time = None
    fall_step = None
    last_used_foot_idx = 0


    for i, ref_seg in enumerate(ref_plan.segments):
        t_ref = ref_seg.t
        x_ref = ref_seg.x
        y_ref = ref_seg.y
        xdot_ref = ref_seg.xdot
        ydot_ref = ref_seg.ydot

        stance_idx = ref_seg.stance_foot_index
        last_used_foot_idx = max(last_used_foot_idx, stance_idx)
        foot_ref = ref_plan.footsteps[stance_idx]
        p_ref_x = foot_ref.x
        p_ref_y = foot_ref.y

        foot_true = true_footsteps[stance_idx]
        p_true_x = foot_true.x
        p_true_y = foot_true.y

        N = len(t_ref)
        if N > 1:
            dt_seg = t_ref[1] - t_ref[0]
            assert np.allclose(np.diff(t_ref), dt_seg, atol=1e-3)

        x_sim = np.zeros(N)
        y_sim = np.zeros(N)
        xdot_sim = np.zeros(N)
        ydot_sim = np.zeros(N)

        for k in range(N):
            x_sim[k] = x
            xdot_sim[k] = xdot
            y_sim[k] = y
            ydot_sim[k] = ydot

            # True stance-relative state
            s_rel_true = np.array([
                x - p_true_x,
                xdot,
                y - p_true_y,
                ydot,
            ])

            # Belief stance-relative state
            s_rel_belief = np.array([
                x - p_ref_x,
                xdot,
                y - p_ref_y,
                ydot,
            ])

            # Reference stance-relative state 
            s_ref_rel = np.array([
                x_ref[k] - p_ref_x,
                xdot_ref[k],
                y_ref[k] - p_ref_y,
                ydot_ref[k],
            ])

            # Error in belief
            e = s_rel_belief - s_ref_rel

            if cl:
                u = -K @ e
            else:
                u = np.zeros(2)

            x_rel_true = s_rel_true[0]
            y_rel_true = s_rel_true[2]
            u_p_max, u_r_max = compute_virtual_inp_lim(x_rel_true, y_rel_true, params)

            u_p = np.clip(u[0], -u_p_max, u_p_max)
            u_r = np.clip(u[1], -u_r_max, u_r_max)
            u = np.array([u_p, u_r])

            max_u = max(max_u, np.max(np.abs(u)))

            # Optional disturbance on virtual inputs
            if dist_fn is not None:
                u = u + dist_fn()

            # Dynamics from true stance frame
            s_rel_next = A_d @ s_rel_true + B_d @ u
            x_rel_next, xdot_next, y_rel_next, ydot_next = s_rel_next

            x = x_rel_next + p_true_x
            y = y_rel_next + p_true_y
            xdot = xdot_next
            ydot = ydot_next

            # Distance from COM to TRUE stance foot
            dist = np.sqrt(x_rel_next**2 + y_rel_next**2)
            max_dist = max(max_dist, dist)

            # Fall condition
            if dist > L_max and not fell:
                fell = True
                fall_time = float(t_ref[k])
                fall_step = i

                # Truncate logs
                x_sim = x_sim[:k + 1]
                y_sim = y_sim[:k + 1]
                xdot_sim = xdot_sim[:k + 1]
                ydot_sim = ydot_sim[:k + 1]
                t_seg = t_ref[:k + 1]

                sim_seg = CoMSegment(
                    phase=ref_seg.phase,
                    t=t_seg.copy(),
                    x=x_sim,
                    xdot=xdot_sim,
                    y=y_sim,
                    ydot=ydot_sim,
                    step_index=ref_seg.step_index,
                    stance_foot_index=ref_seg.stance_foot_index,
                )
                sim_segments.append(sim_seg)
                break 

        if fell:
            break 

        if not fell:
            sim_seg = CoMSegment(
                phase=ref_seg.phase,
                t=t_ref.copy(),
                x=x_sim,
                xdot=xdot_sim,
                y=y_sim,
                ydot=ydot_sim,
                step_index=ref_seg.step_index,
                stance_foot_index=ref_seg.stance_foot_index,
            )
            sim_segments.append(sim_seg)

    used_footsteps = true_footsteps[: last_used_foot_idx + 1]

    sim_plan = GaitPlan(
        footsteps=used_footsteps,
        segments=sim_segments,
    )

    if prnt:
        print(f"Max COM distance to TRUE stance foot: {max_dist:.3f} m")
        print(f"Max virtual input magnitude: {max_u:.3f}")
        if fell:
            print(f"FALL at t = {fall_time:.3f} s, segment index = {fall_step}")
        else:
            print("No fall detected.")

    return sim_plan, fell, fall_time, fall_step