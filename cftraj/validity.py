"""Checks that go beyond ADE/FDE: is the counterfactual response sensible, does
scene context help, and do predicted risk flags match real near-misses?"""

import numpy as np

from .counterfactual import scale_speed
from .metrics import ade_fde
from .model import predict_numpy, predict_windows
from .pairs import all_pairs
from .risk import assess_pair

SPEED_FACTORS = (0.0, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
CONFLICT_RADIUS = 3.0  # meters between centers inside the horizon = conflict
PAIR_RADIUS = 15.0  # only pairs this close at the last observed frame are scored
FLAGGED = ("MEDIUM", "HIGH", "CRITICAL")


def speed_response(model, w, device="cpu", factors=SPEED_FACTORS, n=2000, seed=0):
    """How does the predicted travel distance react to scaling an agent's past speed?

    Travel is the distance from the current position to the predicted final
    position. A physically sensible model should predict more travel for a
    larger factor, and almost none at factor 0. Reports, over ``n`` windows:

    * ``monotone``: fraction whose travel is non-decreasing across ``factors``
    * ``travel_ratio_2x``: mean travel(2.0) / travel(1.0) (1 would mean no
      response; ~2 is a full response, as for constant velocity)
    * ``travel_at_zero_m``: mean predicted travel when the agent was stationary
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(w), size=min(n, len(w)), replace=False)
    last = w.X[idx, -1, :]
    nbr = None if w.nbr is None else w.nbr[idx]
    nmask = None if w.nmask is None else w.nmask[idx]

    travel = []
    for f in factors:
        pred = predict_numpy(model, scale_speed(w.X[idx], f), device, nbr=nbr, nmask=nmask)
        travel.append(np.linalg.norm(pred[:, -1] - last, axis=-1))
    travel = np.stack(travel, axis=1)  # (n, n_factors)

    i1, i2, i0 = factors.index(1.0), factors.index(2.0), factors.index(0.0)
    moving = travel[:, i1] > 0.5  # ignore agents that barely move
    return {
        "monotone": float((np.diff(travel, axis=1) >= -0.05).all(axis=1).mean()),
        "travel_ratio_2x": float((travel[moving, i2] / travel[moving, i1]).mean()),
        "travel_at_zero_m": float(travel[:, i0].mean()),
        "n": len(idx),
    }


def context_ablation(model, w, device="cpu"):
    """ADE/FDE with the real neighbors vs. with all neighbors masked out."""
    if not model.uses_neighbors:
        return None
    with_ctx = ade_fde(predict_windows(model, w, device), w.Y)
    without = ade_fde(predict_windows(model, w, device, mask_neighbors=True), w.Y)
    return {"ade": with_ctx[0], "fde": with_ctx[1],
            "ade_no_context": without[0], "fde_no_context": without[1]}


def conflict_metrics(w, pred, pairs=None, radius=CONFLICT_RADIUS):
    """Precision / recall / F1 of predicted conflicts against real ones.

    Only pairs that are not already within ``radius`` at the last observed frame
    are scored, so the task is predicting *new* approaches. A pair is a *real* conflict if the two agents' true future positions come
    within ``radius`` meters at the same time step. Two predicted rules:

    * ``distance``: predicted time-aligned gap < ``radius``
    * ``risk``: :func:`cftraj.risk.assess_pair` returns MEDIUM or worse
    """
    if pairs is None:
        pairs = all_pairs(w, PAIR_RADIUS)
        current = np.linalg.norm(w.X[pairs[:, 0], -1] - w.X[pairs[:, 1], -1], axis=-1)
        pairs = pairs[current >= radius]  # only *new* approaches, not already-close pairs
    if len(pairs) == 0:
        return {"n_pairs": 0, "n_positive": 0}
    i, j = pairs[:, 0], pairs[:, 1]
    truth = np.linalg.norm(w.Y[i] - w.Y[j], axis=-1).min(axis=1) < radius
    pred_gap = np.linalg.norm(pred[i] - pred[j], axis=-1).min(axis=1)
    rules = {
        "distance": pred_gap < radius,
        "risk": np.array([assess_pair(pred[a], pred[b])["risk"] in FLAGGED
                          for a, b in pairs]),
    }
    out = {"n_pairs": len(pairs), "n_positive": int(truth.sum())}
    for name, flagged in rules.items():
        tp = int((flagged & truth).sum())
        fp = int((flagged & ~truth).sum())
        fn = int((~flagged & truth).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        out[name] = {"precision": p, "recall": r,
                     "f1": 2 * p * r / (p + r) if p + r else 0.0}
    return out
