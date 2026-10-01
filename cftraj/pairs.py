"""Select two agents that actually co-occur in time."""

import numpy as np


def find_interacting_pair(windows, max_distance=None):
    """Return indices ``(i, j)`` of the closest pair of co-occurring agents.

    Two windows co-occur iff they start at the same frame and belong to
    different tracks. "Closest" is measured between the last observed points.
    If ``max_distance`` is given, pairs farther apart are rejected.
    Returns ``None`` when no valid pair exists.
    """
    best, best_d = None, np.inf
    for start in np.unique(windows.starts):
        idx = np.flatnonzero(windows.starts == start)
        if len(idx) < 2:
            continue
        last = windows.X[idx, -1, :]
        d = np.linalg.norm(last[:, None, :] - last[None, :, :], axis=-1)
        d[np.tril_indices(len(idx))] = np.inf  # upper triangle only, no self-pairs
        a, b = np.unravel_index(np.argmin(d), d.shape)
        if d[a, b] < best_d:
            best, best_d = (int(idx[a]), int(idx[b])), float(d[a, b])

    if best is None or (max_distance is not None and best_d > max_distance):
        return None
    return best
