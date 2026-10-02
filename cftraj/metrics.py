import numpy as np


def displacement_errors(pred, true):
    """Euclidean error per window and step: (N, F)."""
    return np.linalg.norm(pred - true, axis=-1)


def ade_fde(pred, true):
    """Average and final displacement error (meters)."""
    err = displacement_errors(pred, true)
    return float(err.mean()), float(err[:, -1].mean())


def error_by_horizon(pred, true):
    """Mean error at each future step: (F,)."""
    return displacement_errors(pred, true).mean(axis=0)


def min_ade_fde(modes, true):
    """Best-of-K ADE/FDE: for each window pick the mode with the lowest ADE.

    ``modes`` is ``(N, K, F, 2)``, ``true`` ``(N, F, 2)``.
    """
    err = np.linalg.norm(modes - true[:, None], axis=-1)  # (N, K, F)
    best = err.mean(-1).argmin(axis=1)
    rows = np.arange(len(modes))
    return float(err[rows, best].mean()), float(err[rows, best, -1].mean())


def bootstrap_ci(values, groups=None, n_boot=1000, alpha=0.05, seed=0):
    """Percentile bootstrap CI of the mean of per-window ``values``.

    With ``groups`` (e.g. a track id per window) whole groups are resampled,
    which respects the strong correlation between windows of one track.
    Returns ``(mean, low, high)``.
    """
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    if groups is None:
        means = [values[rng.integers(0, len(values), len(values))].mean() for _ in range(n_boot)]
    else:
        _, inv = np.unique(np.asarray(groups), return_inverse=True)
        sums = np.bincount(inv, weights=values)
        cnts = np.bincount(inv).astype(float)
        g = len(sums)
        draws = rng.integers(0, g, (n_boot, g))
        means = sums[draws].sum(1) / cnts[draws].sum(1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(values.mean()), float(lo), float(hi)
