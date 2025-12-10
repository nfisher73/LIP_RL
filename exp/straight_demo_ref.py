from include.params import create_default_lip_params
from analytic_src.straight_walk_ref import generate_straight_gait, noisy_feet_gait
from visualization import plot_gait_2d, plot_gait_vel_2d
from visualization import visualize_gait_open3d


params = create_default_lip_params()
plan = generate_straight_gait(params=params)
noisy_plan = noisy_feet_gait(params=params)
plot_gait_2d(plan)
plot_gait_vel_2d(plan)
visualize_gait_open3d(plan, params, swing_height=0.1, frame_interval=0.01)