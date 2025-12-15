from include.params import create_default_lip_params
from analytic_src.straight_walk_sim import straight_walk
from analytic_src.straight_walk_ref import generate_straight_gait
from visualization import plot_2_gaits
from visualization import visualize_gait_open3d
import argparse
from include.disturbance_foos import(
    norm_dist,
    const_x_dist,
    const_y_dist,
    unif_dist,
    unif_x_dist,
    unif_y_dist,
)


def parse_args():
    parser = argparse.ArgumentParser(description="LQR Walk Options")

    parser.add_argument("--dist", type=str, default="norm_dist", help="Disturbance Function")
    parser.add_argument("--scale", type=float, default=None, help="Scale of Disturbance")
    parser.add_argument("--no_viz", action="store_false", help="Toggle Simulation Visual (Default True)")
    parser.add_argument("--no_plot", action="store_false", help="Toggle Trajectory Plot (Default True)")
    parser.add_argument("--ol", action="store_false", help="Toggle Open/Closed Loop (Default Closed Loop)")

    return parser.parse_args()


def get_dist_fn(dist):
    if dist == "norm_dist":
        return norm_dist
    
    if dist == "const_x_dist":
        return const_x_dist
    
    if dist == "const_y_dist":
        return const_y_dist
    
    if dist == "unif_dist":
        return unif_dist
    
    if dist == "unif_x_dist":
        return unif_x_dist
    
    if dist == "unif_y_dist":
        return unif_y_dist
    
    raise ValueError("Disturbance Function not a valid option.")
    


def main():
    args = parse_args()
    params = create_default_lip_params()
    ref_plan = generate_straight_gait(params)
    dist = get_dist_fn(args.dist)
    walk = straight_walk(params=params, cl = args.ol, dist_fn = dist, scale = args.scale)
    if args.no_plot:
        plot_2_gaits(ref_plan, title = "Simulated LIP Walking Trajectory", extra_segments = walk.segments, extra_label = "CoM Position")
    if args.no_viz:
        visualize_gait_open3d(walk, params, swing_height=0.1, frame_interval=0.01)
    return None



if __name__ == "__main__":
    main()