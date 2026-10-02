"""Select two agents that actually co-occur in time."""

import numpy as np

from .data import group_indices


def find_interacting_pair(windows, max_distance=None):
    """Return indices ``(i, j)`` of the closest pair of co-occurring agents.

    Two windows co-occur iff they are in the same sequence, start at the same
    frame and belong to different tracks. "Closest" is measured between the
    last observed points (meters). If ``max_distance`` is given, a pair
    farther apart is rejected. Returns ``None`` when no valid pair exists.
    """
    best, best_d = None, np.inf
    for idx in group_indices(windows.group_keys()):
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


def all_pairs(windows, max_distance=15.0):
    """All co-occurring window pairs ``(i, j)``, ``i < j``, within ``max_distance`` m."""
    out = []
    for idx in group_indices(windows.group_keys()):
        if len(idx) < 2:
            continue
        last = windows.X[idx, -1, :]
        d = np.linalg.norm(last[:, None, :] - last[None, :, :], axis=-1)
        a, b = np.nonzero(np.triu(d <= max_distance, k=1))
        out.extend(zip(idx[a].tolist(), idx[b].tolist()))
    return np.array(out, dtype=np.int64).reshape(-1, 2)


def neighbor_slots(windows, pairs):
    """For each pair ``(i, j)``, the slot of agent ``i`` among window ``j``'s neighbors.

    ``-1`` where ``i`` is not among ``j``'s (nearest ``K``) neighbors.
    """
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    match = windows.ntids[pairs[:, 1]] == windows.tids[pairs[:, 0]][:, None]  # (n, K)
    return np.where(match.any(axis=1), match.argmax(axis=1), -1)


def neighbors_with_replaced(windows, pairs, new_pasts):
    """``nbr`` rows of each pair's second window, with agent ``i``'s slot showing ``new_pasts``.

    ``new_pasts`` is ``(n, P, 2)``. Pairs where ``i`` is not a neighbor are
    returned unchanged. Output ``(n, K, P, 2)``.
    """
    pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
    nbr = windows.nbr[pairs[:, 1]].copy()
    slots = neighbor_slots(windows, pairs)
    ok = np.flatnonzero(slots >= 0)
    nbr[ok, slots[ok]] = new_pasts[ok]
    return nbr
