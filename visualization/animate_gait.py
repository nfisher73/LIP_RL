import open3d as o3d
import numpy as np
import time
from .animation_utils import build_gait_frames



def visualize_gait_open3d(gait_plan, params, swing_height=0.1, frame_interval=0.03):
    """
    Visualize a GaitPlan in Open3D as a stick robot with a floor:
    - Sphere for CoM
    - Spheres for left/right feet
    - Line segments for legs
    - Gray floor at z = 0
    - Camera follows the CoM so the robot stays in view
    """
    z_c = params.z_c
    frames = build_gait_frames(gait_plan, z_c=z_c, swing_height=swing_height)

    # --- Create geometries ---

    # CoM sphere
    com_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
    com_mesh.compute_vertex_normals()
    com_mesh.paint_uniform_color([1.0, 0.0, 0.0])  # red

    # Feet spheres
    left_foot_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.025)
    left_foot_mesh.compute_vertex_normals()
    left_foot_mesh.paint_uniform_color([0.0, 0.0, 1.0])  # blue

    right_foot_mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.025)
    right_foot_mesh.compute_vertex_normals()
    right_foot_mesh.paint_uniform_color([0.0, 1.0, 0.0])  # green

    # Initial positions = first frame
    com_pos0, left0, right0 = frames[0]
    com_mesh.translate(com_pos0)
    left_foot_mesh.translate(left0)
    right_foot_mesh.translate(right0)

    # LineSets for legs (two points: CoM and each foot)
    left_leg = o3d.geometry.LineSet()
    left_leg.points = o3d.utility.Vector3dVector(
        np.vstack([com_pos0, left0])
    )
    left_leg.lines = o3d.utility.Vector2iVector([[0, 1]])
    left_leg.colors = o3d.utility.Vector3dVector([[0.0, 0.0, 0.0]])  # black

    right_leg = o3d.geometry.LineSet()
    right_leg.points = o3d.utility.Vector3dVector(
        np.vstack([com_pos0, right0])
    )
    right_leg.lines = o3d.utility.Vector2iVector([[0, 1]])
    right_leg.colors = o3d.utility.Vector3dVector([[0.0, 0.0, 0.0]])

    # --- Floor (large thin box at z=0) ---
    floor_length = 7.5   # along x
    floor_width  = 1.5    # along y
    floor_thick  = 0.02   # along z

    floor = o3d.geometry.TriangleMesh.create_box(
        width=floor_length,
        height=floor_width,
        depth=floor_thick,
    )
    floor.compute_vertex_normals()
    floor.paint_uniform_color([0.85, 0.85, 0.85])  # light gray

    # Place floor so its top surface is at z = 0 and centered near origin
    # Open3D box is [0, width] x [0, height] x [0, depth]
    floor.translate(np.array([-1, -floor_width / 2, -floor_thick]))

    # --- Set up visualizer ---
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="LIP Stick Walker", width=960, height=720)
    opt = vis.get_render_option()
    # opt.light_on = False                     # disable shading
    opt.background_color = np.array([1, 1, 1])  # optional: white background



    wait_flag = {"start": False, "done": False}

    def on_space_start(vis_inner):
        wait_flag["start"] = True
        return False  # don't close window automatically

    def on_space_done(vis_inner):
        wait_flag["done"] = True
        return False

    # Initially, SPACE means "start animation"
    vis.register_key_callback(32, on_space_start)

    vis.add_geometry(floor)
    vis.add_geometry(com_mesh)
    vis.add_geometry(left_foot_mesh)
    vis.add_geometry(right_foot_mesh)
    vis.add_geometry(left_leg)
    vis.add_geometry(right_leg)

    # Camera control
    ctr = vis.get_view_control()

    def set_camera(com_pos):
        """
        Third-person camera: above and behind the robot, looking forward
        along +x with a slight downward angle.
        """
        # Point we want to look at: a bit above the CoM
        center = com_pos + np.array([0.0, 0.0, 0.15])

        # Choose where the camera should *roughly* be relative to the CoM.
        # Negative x = behind, negative y = slightly to the right,
        # positive z = above.
        cam_offset = np.array([-1.8, -1.0, -1.0])

        # front is the viewing direction: from camera to lookat.
        # Since "eye = center + something", vector from eye to center is -cam_offset.
        front = -cam_offset
        front = front / np.linalg.norm(front)

        up = np.array([0.0, 0.0, 1.0])

        ctr.set_lookat(center.tolist())
        ctr.set_front(front.tolist())
        ctr.set_up(up.tolist())
        ctr.set_zoom(0.1)  # larger = closer in; tweak between ~0.7–1.0 if needed

    # Initial camera
    set_camera(com_pos0)

    # --- Wait for SPACE to start animation ---
    print("Press SPACE to start animation...")
    while not wait_flag["start"]:
        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.05)

    # Track current positions so we can translate by deltas
    current_com   = com_pos0.copy()
    current_left  = left0.copy()
    current_right = right0.copy()

    # --- Animation loop ---
    for (com_pos, left_pos, right_pos) in frames[1:]:
        # Move CoM sphere
        com_mesh.translate(com_pos - current_com, relative=True)
        current_com = com_pos

        # Move feet spheres
        left_foot_mesh.translate(left_pos - current_left, relative=True)
        right_foot_mesh.translate(right_pos - current_right, relative=True)
        current_left  = left_pos
        current_right = right_pos

        # Update leg line endpoints
        left_leg.points = o3d.utility.Vector3dVector(
            np.vstack([com_pos, left_pos])
        )
        right_leg.points = o3d.utility.Vector3dVector(
            np.vstack([com_pos, right_pos])
        )

        # Update camera to follow
        #set_camera(com_pos)
        ctr.set_lookat((com_pos + np.array([0, 0, 0.15])).tolist())


        vis.update_geometry(com_mesh)
        vis.update_geometry(left_foot_mesh)
        vis.update_geometry(right_foot_mesh)
        vis.update_geometry(left_leg)
        vis.update_geometry(right_leg)
        vis.update_geometry(floor)

        vis.poll_events()
        vis.update_renderer()

        time.sleep(frame_interval)

    # --- After animation: wait for SPACE to close ---
    # Rebind SPACE to "done"
    vis.register_key_callback(32, on_space_done)

    print("Animation finished. Press SPACE to close.")
    while not wait_flag["done"]:
        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.05)

    vis.destroy_window()

