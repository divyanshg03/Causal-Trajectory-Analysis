"""Counterfactual interventions on an agent's observed past.

Both interventions rewrite the *history* of one agent while keeping its
current (last) position fixed, so the question asked is "what would the model
predict had this agent been moving differently up to now?". Inputs may carry
leading batch dimensions: ``(..., P, 2)``.
"""

import numpy as np


def scale_speed(traj, factor=1.5, anchor="last"):
    """Re-integrate ``traj`` with every per-step displacement scaled by ``factor``.

    ``factor > 1`` means the agent had been moving faster, ``0 <= factor < 1``
    slower, ``0`` stationary. ``anchor="last"`` keeps the *current* (last)
    position fixed, so only the agent's motion history changes, not where it
    is now; ``anchor="first"`` keeps the first point fixed instead.
    """
    if anchor not in ("first", "last"):
        raise ValueError("anchor must be 'first' or 'last'")
    traj = np.asarray(traj, dtype=np.float32)
    out = np.empty_like(traj)
    out[..., 0, :] = traj[..., 0, :]
    out[..., 1:, :] = traj[..., :1, :] + np.cumsum(np.diff(traj, axis=-2) * factor, axis=-2)
    if anchor == "last":
        out += traj[..., -1:, :] - out[..., -1:, :]
    return out


def lateral_shift(traj, offset):
    """Make the agent look like it drifted ``offset`` meters sideways (x) to reach
    its current position (a lane-change-like history). Current position is kept."""
    traj = np.asarray(traj, dtype=np.float32)
    ramp = np.linspace(-1.0, 0.0, traj.shape[-2], dtype=np.float32)  # -1 at start, 0 now
    out = traj.copy()
    out[..., 0] += offset * ramp
    return out


INTERVENTIONS = ("speed", "lateral")


def apply_intervention(past, kind="speed", value=2.0):
    """Dispatch: ``speed`` (value = speed factor) or ``lateral`` (value = meters)."""
    if kind == "speed":
        return scale_speed(past, value)
    if kind == "lateral":
        return lateral_shift(past, value)
    raise ValueError(f"Unknown intervention '{kind}'. Choose from {INTERVENTIONS}")
