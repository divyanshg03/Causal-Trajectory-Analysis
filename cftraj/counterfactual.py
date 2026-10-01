"""Counterfactual interventions on an agent's observed past."""

import numpy as np


def scale_speed(traj, factor=1.5):
    """Re-integrate ``traj`` with every per-step displacement scaled by ``factor``.

    The first point is kept fixed. ``factor > 1`` speeds the agent up,
    ``0 <= factor < 1`` slows it down, ``factor == 0`` stops it.
    """
    traj = np.asarray(traj, dtype=np.float32)
    steps = np.diff(traj, axis=0) * factor
    out = np.empty_like(traj)
    out[0] = traj[0]
    out[1:] = traj[0] + np.cumsum(steps, axis=0)
    return out
