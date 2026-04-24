import numpy as np

def compute_velocity(traj):
    return traj[-1] - traj[-2]


def compute_distance(traj1, traj2):
    return np.linalg.norm(traj1[-1] - traj2[-1])


def compute_relation(traj1, traj2):
    pos_diff = traj2[-1] - traj1[-1]

    vel1 = compute_velocity(traj1)
    vel2 = compute_velocity(traj2)

    rel_vel = vel2 - vel1

    dot = np.dot(pos_diff, rel_vel)

    if dot < 0:
        return "approaching"
    elif dot > 0:
        return "moving_away"
    else:
        return "parallel"

def compute_ttc(distance, motion):
    if motion >= 0:
        return float('inf')
    return distance / (abs(motion) + 1e-6)


def compute_risk(distance, motion):
    ttc = compute_ttc(distance, motion)

    if ttc < 1:
        return "CRITICAL"
    elif ttc < 3:
        return "HIGH"
    elif ttc < 5:
        return "MEDIUM"
    elif distance < 0.02:
        return "LOW"
    else:
        return "SAFE"
    
def build_scene_graph(traj1, traj2):
    distance = compute_distance(traj1, traj2)

    relation = compute_relation(traj1, traj2)

    # compute motion SAME WAY as main pipeline
    pos_diff = traj2[-1] - traj1[-1]
    vel1 = compute_velocity(traj1)
    vel2 = compute_velocity(traj2)
    vel_diff = vel2 - vel1

    norm = np.linalg.norm(pos_diff) + 1e-6
    motion = np.dot(pos_diff, vel_diff) / norm

    risk = compute_risk(distance, motion)

    return {
        "object_A": "Car_A",
        "object_B": "Car_B",
        "relation": relation,
        "distance": float(distance),
        "motion": float(motion),
        "risk": risk
    }