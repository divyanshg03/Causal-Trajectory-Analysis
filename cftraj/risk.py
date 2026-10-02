"""Pairwise interaction metrics, TTC-based risk and scene-graph construction.

Geometry is in **meters** (BEV, ego-camera frame) and velocities are per
frame, so times are in seconds at the KITTI frame rate. Everything uses the
*relative* motion of the two predicted trajectories, which is unaffected by
the ego camera moving.

Time to collision is measured from "now" (the last observed frame):

* if the predicted gap drops below ``COLLISION_RADIUS`` inside the horizon,
  TTC is the time of the first such step;
* otherwise, if the agents are still approaching at the end of the horizon,
  TTC extrapolates that closing speed from the horizon end;
* otherwise it is infinite.
"""

import numpy as np

from .data import KITTI_FPS

EPS = 1e-6

COLLISION_RADIUS = 2.0  # meters between agent centers counted as overlap
NEAR_DISTANCE = 5.0  # meters; "LOW" if the pair ever comes this close
# (upper TTC bound in seconds, label); evaluated in order.
TTC_BANDS = ((1.0, "CRITICAL"), (3.0, "HIGH"), (5.0, "MEDIUM"))
RISK_ORDER = ("SAFE", "LOW", "MEDIUM", "HIGH", "CRITICAL")


def distance(traj1, traj2):
    """Euclidean distance between the last points of two trajectories."""
    return float(np.linalg.norm(traj1[-1] - traj2[-1]))


def gap_series(traj1, traj2):
    """Time-aligned distance between the two trajectories, shape (T,)."""
    n = min(len(traj1), len(traj2))
    return np.linalg.norm(np.asarray(traj2[:n]) - np.asarray(traj1[:n]), axis=1)


def min_distance(traj1, traj2):
    return float(gap_series(traj1, traj2).min())


def closing_speed(traj1, traj2, k=3):
    """Rate (m/frame) at which agent 2 moves towards agent 1 at the last step.

    Velocity is a ``k``-step average to damp prediction jitter. Negative means
    the agents are approaching each other.
    """
    n = min(len(traj1), len(traj2))
    k = max(1, min(k, n - 1))
    rel = np.asarray(traj2[:n]) - np.asarray(traj1[:n])
    rel_vel = (rel[-1] - rel[-1 - k]) / k
    return float(np.dot(rel[-1], rel_vel) / (np.linalg.norm(rel[-1]) + EPS))


def relation(traj1, traj2):
    """Overall trend of the gap over the horizon."""
    gap = gap_series(traj1, traj2)
    delta = gap[-1] - gap[0]
    if delta < -0.05:
        return "approaching"
    if delta > 0.05:
        return "moving_away"
    return "parallel"


def time_to_collision(traj1, traj2, fps=KITTI_FPS, radius=COLLISION_RADIUS):
    """Time to collision in seconds from the last observed frame (see module doc)."""
    gap = gap_series(traj1, traj2)
    hit = np.flatnonzero(gap < radius)
    if len(hit):
        return float(hit[0] + 1) / fps
    closing = closing_speed(traj1, traj2)
    if closing >= 0:
        return float("inf")
    return float(len(gap) / fps + max(gap[-1] - radius, 0.0) / abs(closing) / fps)


def risk_from_ttc(ttc, min_dist, near_distance=NEAR_DISTANCE):
    for bound, label in TTC_BANDS:
        if ttc < bound:
            return label
    return "LOW" if min_dist < near_distance else "SAFE"


def assess_pair(traj1, traj2, fps=KITTI_FPS, radius=COLLISION_RADIUS):
    """Risk summary of two predicted trajectories (each ``(T, 2)`` in meters)."""
    min_d = min_distance(traj1, traj2)
    ttc = time_to_collision(traj1, traj2, fps, radius)
    return {
        "relation": relation(traj1, traj2),
        "distance_m": distance(traj1, traj2),
        "min_distance_m": min_d,
        "closing_speed_mps": closing_speed(traj1, traj2) * fps,
        "ttc_s": ttc,
        "risk": risk_from_ttc(ttc, min_d),
    }


def build_scene_graph(traj1, traj2, name_a="Car_A", name_b="Car_B"):
    """Describe the relation between two predicted trajectories."""
    return {"object_A": name_a, "object_B": name_b, **assess_pair(traj1, traj2)}


def explain(scene):
    a, b = scene["object_A"], scene["object_B"]
    if np.isfinite(scene["ttc_s"]):
        verb = "are on course to meet" if scene["ttc_s"] < 10 else "are slowly closing in"
        return (
            f"{a} and {b} {verb}: closest approach "
            f"{scene['min_distance_m']:.1f} m, TTC={scene['ttc_s']:.1f} s. "
            f"Risk: {scene['risk']}."
        )
    verb = "moving away from" if scene["relation"] == "moving_away" else "keeping pace with"
    return (
        f"{a} is {verb} {b} (closest approach {scene['min_distance_m']:.1f} m). "
        f"Risk: {scene['risk']}."
    )


def risk_distribution(samples_a, samples_b, fps=KITTI_FPS, radius=COLLISION_RADIUS):
    """Risk under prediction uncertainty from ``S`` sampled futures per agent.

    ``samples_a`` / ``samples_b`` are ``(S, T, 2)`` (sample ``s`` of A is paired
    with sample ``s`` of B). Returns the probability of a predicted overlap
    (gap < ``radius``), of MEDIUM-or-worse risk, the mean/quantile TTC and the
    fraction of samples in each tier.
    """
    reports = [assess_pair(a, b, fps, radius) for a, b in zip(samples_a, samples_b)]
    ttc = np.array([r["ttc_s"] for r in reports])
    labels = [r["risk"] for r in reports]
    overlap = np.array([r["min_distance_m"] < radius for r in reports])
    return {
        "n_samples": len(reports),
        "p_overlap": float(overlap.mean()),
        "p_flagged": float(np.mean([RISK_ORDER.index(lab) >= RISK_ORDER.index("MEDIUM")
                                    for lab in labels])),
        "ttc_median_s": float(np.quantile(ttc, 0.5, method="lower")),
        "ttc_p10_s": float(np.quantile(ttc, 0.1, method="lower")),
        "tier_fractions": {t: float(np.mean([lab == t for lab in labels])) for t in RISK_ORDER},
    }


def calibrate_ttc_threshold(ttc, truth, candidates=None):
    """TTC threshold (s) that maximises F1 for flagging real conflicts.

    ``ttc`` are predicted TTCs (inf allowed) and ``truth`` booleans. Returns
    ``(threshold, f1)``; a pair is flagged when ``ttc < threshold``.
    """
    ttc = np.asarray(ttc, dtype=float)
    truth = np.asarray(truth, dtype=bool)
    if candidates is None:
        candidates = np.arange(0.5, 10.01, 0.5)
    best_t, best_f1 = float(candidates[0]), -1.0
    for t in candidates:
        flag = ttc < t
        tp = int((flag & truth).sum())
        fp = int((flag & ~truth).sum())
        fn = int((~flag & truth).sum())
        f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0
        if f1 > best_f1:
            best_t, best_f1 = float(t), f1
    return best_t, best_f1


def tier_reliability(labels, truth):
    """Reliability table: for each tier, how often a real conflict followed it.

    Returns ``{tier: {"n": count, "conflict_rate": fraction or None}}``. A
    useful tiering has a conflict rate that increases with severity.
    """
    labels = np.asarray(labels)
    truth = np.asarray(truth, dtype=bool)
    out = {}
    for tier in RISK_ORDER:
        sel = labels == tier
        out[tier] = {"n": int(sel.sum()),
                     "conflict_rate": float(truth[sel].mean()) if sel.any() else None}
    return out
