import numpy as np

def build_gait_frames(gait_plan, z_c=0.8, swing_height=0.1):
    """
    Turn a GaitPlan into a list of frames.
    Each frame: (com_pos, left_foot_pos, right_foot_pos)
      where each is a (3,) np.array [x, y, z].
    """
    footsteps = gait_plan.footsteps
    segments = gait_plan.segments

    # Convenience: store indexes of left/right footsteps
    left_indices  = [i for i, f in enumerate(footsteps) if f.is_left]
    right_indices = [i for i, f in enumerate(footsteps) if not f.is_left]

    frames = []

    def find_prev_next_foot_indices(is_left, stance_index):
        """For the swing leg: find previous and next footholds for that leg."""
        indices = left_indices if is_left else right_indices
        # Previous foothold (index < stance_index)
        prev_candidates = [i for i in indices if i < stance_index]
        next_candidates = [i for i in indices if i > stance_index]

        prev_idx = prev_candidates[-1] if prev_candidates else None
        next_idx = next_candidates[0] if next_candidates else None
        return prev_idx, next_idx

    for seg in segments:
        t_seg = np.asarray(seg.t)
        x_seg = np.asarray(seg.x)
        y_seg = np.asarray(seg.y)
        n_samples = len(t_seg)

        stance_idx = seg.stance_foot_index
        stance_foot = footsteps[stance_idx]
        stance_is_left = stance_foot.is_left
        swing_is_left = not stance_is_left

        # Get stance foot pos (constant over this segment)
        stance_pos = np.array([stance_foot.x, stance_foot.y, 0.0])

        # For the swing leg: find previous and next footholds
        prev_idx, next_idx = find_prev_next_foot_indices(swing_is_left, stance_idx)

        if prev_idx is None:
            # No previous foothold for this leg (very first step): keep at its first known foothold
            # We'll just keep swing leg fixed in this segment.
            if next_idx is None:
                swing_start = swing_target = stance_pos.copy()
            else:
                swing_start = swing_target = np.array([
                    footsteps[next_idx].x,
                    footsteps[next_idx].y,
                    0.0,
                ])
        else:
            swing_start = np.array([
                footsteps[prev_idx].x,
                footsteps[prev_idx].y,
                0.0,
            ])
            if next_idx is not None:
                swing_target = np.array([
                    footsteps[next_idx].x,
                    footsteps[next_idx].y,
                    0.0,
                ])
            else:
                # No future step for this leg: keep at last foothold
                swing_target = swing_start.copy()

        # For each time sample in this segment, compute positions
        for k in range(n_samples):
            # CoM
            com_pos = np.array([x_seg[k], y_seg[k], z_c])

            # Normalized phase in this step
            if n_samples > 1:
                s = k / (n_samples - 1)
            else:
                s = 0.0

            # Stance foot stays flat on the ground
            stance_3d = stance_pos.copy()

            # Swing foot interpolates from start to target, with a height bump
            if np.allclose(swing_start, swing_target):
                # No movement for this leg in this segment
                swing_xy = swing_start.copy()
                swing_z = 0.0
            else:
                swing_xy = (1.0 - s) * swing_start + s * swing_target
                swing_z = 4.0 * swing_height * s * (1.0 - s)  # parabola peak at s=0.5

            swing_3d = swing_xy.copy()
            swing_3d[2] = swing_z

            # Assign to left/right
            if stance_is_left:
                left_foot_pos  = stance_3d
                right_foot_pos = swing_3d
            else:
                right_foot_pos = stance_3d
                left_foot_pos  = swing_3d

            frames.append((com_pos, left_foot_pos, right_foot_pos))

    return frames
