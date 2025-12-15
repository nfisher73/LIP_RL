from include.params import create_default_lip_params
from analytic_src.straight_walk_sim import straight_walk, straight_walk_with_foot_errors, straight_walk_with_start_foot_error
from visualization import plot_gait_2d, plot_gait_vel_2d
from visualization import visualize_gait_open3d
from include.disturbance_foos import(
    rand_dist,
    const_x_dist,
    const_y_dist,
    unif_dist,
    unif_x_dist,
    unif_y_dist,
)
import numpy as np


params = create_default_lip_params()
# ol_walk = straight_walk(params=params, cl = False, dist_fn = None)

# cl_walk, fell, fall_time, fall_step = straight_walk_with_foot_errors(params, dist_fn = unif_dist, foot_noise_std=0)
# plot_gait_2d(cl_walk)

# if fell: 
#     visualize_gait_open3d(cl_walk, params, swing_height=0.1, frame_interval=0.01)

# cl_walk, fell, fall_time, fall_step = straight_walk_with_foot_errors(params, dist_fn = unif_dist, foot_noise_std=0.005)
# plot_gait_2d(cl_walk)

# if fell: 
#     visualize_gait_open3d(cl_walk, params, swing_height=0.1, frame_interval=0.01)
#fall_steps = []
#for i in range(1000):
cl_walk, fell, fall_time, fall_step = straight_walk_with_foot_errors(params, dist_fn = None, foot_noise_std=0.014, prnt= False, cl = True)
visualize_gait_open3d(cl_walk, params, swing_height=0.1, frame_interval=0.02)
#     fall_steps.append(fall_step)
#     print(cl_walk)
#     break
# print(np.mean(fall_steps))
# plot_gait_vel_2d(cl_walk)
#visualize_gait_open3d(cl_walk, params, swing_height=0.1, frame_interval=0.02)

# cl_walk, fell, fall_time, fall_step = straight_walk_with_foot_errors(params, dist_fn = unif_dist, foot_noise_std=0.015)
# plot_gait_2d(cl_walk)

# cl_walk, fell, fall_time, fall_step = straight_walk_with_start_foot_error(params, dist_fn = unif_dist, foot_noise_std=0.022)
# plot_gait_2d(cl_walk)

# if fell: 
#     visualize_gait_open3d(cl_walk, params, swing_height=0.1, frame_interval=0.01)