from include.params import create_default_lip_params
from analytic_src.straight_walk_sim import straight_walk, noisy_straight_walk
from analytic_src.straight_walk_ref import generate_straight_gait
from visualization import plot_gait_2d, plot_gait_vel_2d, plot_2_gaits
from visualization import visualize_gait_open3d
import numpy as np
from include.disturbance_foos import(
    rand_dist,
    const_x_dist,
    const_y_dist,
    unif_dist,
    unif_x_dist,
    unif_y_dist,
)


params = create_default_lip_params()
# ol_walk = straight_walk(params=params, cl = False, dist_fn = None)

# cl_walk = noisy_straight_walk(params=params)
# plot_gait_2d(cl_walk)

# cl_walk = noisy_straight_walk(params=params, noise_std = 1)
# plot_gait_2d(cl_walk)
# visualize_gait_open3d(cl_walk, params, swing_height=0.1, frame_interval=0.01)
ref_plan = generate_straight_gait(params)
cl_walk = straight_walk(params=params, cl = True, dist_fn = rand_dist)
plot_2_gaits(ref_plan, title= "LIP CL Walking Trajectory with Disturbances", extra_segments = cl_walk.segments, extra_label = "CoM Positions")

# cl_walk = straight_walk(params=params, cl = True, dist_fn = const_y_dist)
# plot_gait_2d(cl_walk)
#visualize_gait_open3d(cl_walk, params, swing_height=0.1, frame_interval=0.01)