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
