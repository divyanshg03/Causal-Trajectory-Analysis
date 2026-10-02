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


def brake(traj, strength=0.8):
    """Make the agent look like it has been braking: per-step speed ramps linearly from
    its original value down to ``(1 - strength)`` of it at the current frame
    (``strength=1`` = coming to a full stop now). Current position is kept."""
    if not 0.0 <= strength <= 1.0:
        raise ValueError("brake strength must be in [0, 1]")
    traj = np.asarray(traj, dtype=np.float32)
    steps = np.diff(traj, axis=-2)
    ramp = np.linspace(1.0, 1.0 - strength, steps.shape[-2], dtype=np.float32)[:, None]
    out = np.empty_like(traj)
    out[..., 0, :] = 0.0
    out[..., 1:, :] = np.cumsum(steps * ramp, axis=-2)
    return out + (traj[..., -1:, :] - out[..., -1:, :])


def delay_start(traj, frames=5):
    """Make the agent look like it only started moving ``frames`` frames ago: it sits at its
    first position and the original motion is replayed over the remaining steps
    (same average pace as the shortened history). Current position is kept."""
    traj = np.asarray(traj, dtype=np.float32)
    n = traj.shape[-2]
    frames = int(round(frames))
    if not 0 <= frames < n - 1:
        raise ValueError(f"delay frames must be in [0, {n - 2}]")
    if frames == 0:
        return traj.copy()
    moving = n - 1 - frames
    # replay the whole original path (linearly resampled) over the last `moving` steps
    pos = np.linspace(0.0, n - 1, moving + 1)
    lo = np.minimum(pos.astype(int), n - 2)
    frac = (pos - lo).astype(np.float32)[:, None]
    path = traj[..., lo, :] * (1 - frac) + traj[..., lo + 1, :] * frac
    out = np.empty_like(traj)
    out[..., : frames + 1, :] = traj[..., :1, :]
    out[..., frames:, :] = path
    return out + (traj[..., -1:, :] - out[..., -1:, :])


INTERVENTIONS = ("speed", "lateral", "brake", "delay")


def apply_intervention(past, kind="speed", value=2.0):
    """Dispatch: ``speed`` (value = speed factor), ``lateral`` (meters), ``brake``
    (strength in [0, 1]) or ``delay`` (frames the agent sat still)."""
    if kind == "speed":
        return scale_speed(past, value)
    if kind == "lateral":
        return lateral_shift(past, value)
    if kind == "brake":
        return brake(past, value)
    if kind == "delay":
        return delay_start(past, value)
    raise ValueError(f"Unknown intervention '{kind}'. Choose from {INTERVENTIONS}")


MAX_ACCEL = 10.0  # m/s^2; beyond this a road user's history is not physically plausible
FPS = 10.0


def max_acceleration(traj, fps=FPS):
    """Peak acceleration magnitude (m/s^2) of ``(..., P, 2)`` trajectories."""
    traj = np.asarray(traj, dtype=np.float32)
    acc = np.diff(traj, n=2, axis=-2) * fps**2
    return np.linalg.norm(acc, axis=-1).max(axis=-1)


def is_plausible(traj, max_accel=MAX_ACCEL, fps=FPS):
    """True where a (possibly intervened) history stays under the acceleration limit."""
    return max_acceleration(traj, fps) <= max_accel
