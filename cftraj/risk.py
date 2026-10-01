"""Pairwise interaction metrics, TTC-based risk and scene-graph construction.

All geometry here is in *normalized image-plane* units (see ``data.normalize``)
and velocities are per frame. TTC is therefore reported in frames and
converted to seconds with the KITTI frame rate; it is an image-plane
approximation, not a metric time-to-collision.
"""

import numpy as np

KITTI_FPS = 10.0
EPS = 1e-6

# (upper TTC bound in seconds, label); evaluated in order.
TTC_BANDS = ((1.0, "CRITICAL"), (3.0, "HIGH"), (5.0, "MEDIUM"))
NEAR_DISTANCE = 0.02  # normalized units; "LOW" if closer than this and not approaching fast


def velocity(traj):
    return traj[-1] - traj[-2]


def distance(traj1, traj2):
    """Euclidean distance between the last points of two trajectories."""
    return float(np.linalg.norm(traj1[-1] - traj2[-1]))


def min_distance(traj1, traj2):
    """Minimum distance over time-aligned points."""
    n = min(len(traj1), len(traj2))
    return float(np.min(np.linalg.norm(traj1[:n] - traj2[:n], axis=1)))


def closing_speed(traj1, traj2):
    """Rate (units/frame) at which agent 2 moves towards agent 1.

    Negative means the agents are approaching each other.
    """
    pos_diff = traj2[-1] - traj1[-1]
    vel_diff = velocity(traj2) - velocity(traj1)
    return float(np.dot(pos_diff, vel_diff) / (np.linalg.norm(pos_diff) + EPS))


def relation(motion):
    if motion < 0:
        return "approaching"
    if motion > 0:
        return "moving_away"
    return "parallel"


def ttc_seconds(dist, motion, fps=KITTI_FPS):
    """Time to collision in seconds (``inf`` unless approaching)."""
    if motion >= 0:
        return float("inf")
    return dist / abs(motion) / fps


def classify_risk(dist, motion, fps=KITTI_FPS):
    ttc = ttc_seconds(dist, motion, fps)
    for bound, label in TTC_BANDS:
        if ttc < bound:
            return label
    return "LOW" if dist < NEAR_DISTANCE else "SAFE"


def build_scene_graph(traj1, traj2, name_a="Car_A", name_b="Car_B"):
    """Describe the relation between two predicted trajectories."""
    dist = distance(traj1, traj2)
    motion = closing_speed(traj1, traj2)
    return {
        "object_A": name_a,
        "object_B": name_b,
        "relation": relation(motion),
        "distance": dist,
        "motion": motion,
        "ttc_s": ttc_seconds(dist, motion),
        "risk": classify_risk(dist, motion),
    }


def explain(scene):
    a, b = scene["object_A"], scene["object_B"]
    if scene["motion"] < 0:
        return (
            f"{a} is approaching {b} (distance={scene['distance']:.4f}, "
            f"TTC={scene['ttc_s']:.2f}s). Risk: {scene['risk']}."
        )
    return (
        f"{a} is moving away from {b} (distance={scene['distance']:.4f}). "
        f"Risk: {scene['risk']}."
    )
