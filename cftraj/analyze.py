"""End-to-end counterfactual collision analysis for one pair of agents."""

import numpy as np
import torch

from .counterfactual import apply_intervention
from .pairs import all_pairs, find_interacting_pair
from .risk import RISK_ORDER, build_scene_graph, explain


@torch.no_grad()
def predict_one(model, past, nbr=None, nmask=None):
    """Absolute future ``(F, 2)`` in meters for one absolute past ``(P, 2)``."""
    model.eval()
    x = torch.as_tensor(np.asarray(past), dtype=torch.float32).unsqueeze(0)
    nb = nm = None
    if model.uses_neighbors and nbr is not None:
        nb = torch.as_tensor(np.asarray(nbr), dtype=torch.float32).unsqueeze(0)
        nm = torch.as_tensor(np.asarray(nmask), dtype=torch.bool).unsqueeze(0)
    out = model(x, nb, nm).numpy()[0]
    attn = None
    if model.uses_neighbors and nbr is not None:
        attn = model.last_attn[0].numpy().copy()
    return out, attn


def _replace_neighbor(nbr, ntids, tid, new_past):
    """Copy of ``nbr`` where the slot holding track ``tid`` shows ``new_past``."""
    nbr = nbr.copy()
    slots = np.flatnonzero(ntids == tid)
    if len(slots):
        nbr[slots[0]] = new_past
    return nbr


def analyze_pair(model, windows, kind="speed", value=2.0, max_distance=None, pair=None,
                 mask_neighbors=False):
    """Compare predicted risk with and without an intervention on agent A.

    The intervention rewrites A's observed past. For a social model, B sees the
    intervened A as a neighbor, so both predictions react. With
    ``mask_neighbors`` the model is run without any scene context.
    Returns ``None`` if the windows contain no co-occurring pair.
    """
    pair = pair if pair is not None else find_interacting_pair(windows, max_distance)
    if pair is None:
        return None
    i, j = pair
    past_a, past_b = windows.X[i], windows.X[j]
    use_ctx = model.uses_neighbors and windows.nbr is not None
    nbr_a = nbr_b = nm_a = nm_b = None
    if use_ctx:
        nbr_a, nbr_b = windows.nbr[i], windows.nbr[j]
        nm_a, nm_b = windows.nmask[i], windows.nmask[j]
        if mask_neighbors:
            nm_a, nm_b = np.zeros_like(nm_a), np.zeros_like(nm_b)

    pred_a, attn_a = predict_one(model, past_a, nbr_a, nm_a)
    pred_b, attn_b = predict_one(model, past_b, nbr_b, nm_b)

    past_a_cf = apply_intervention(past_a, kind, value)
    nbr_b_cf = (
        _replace_neighbor(nbr_b, windows.ntids[j], windows.tids[i], past_a_cf) if use_ctx else None
    )
    cf_a, _ = predict_one(model, past_a_cf, nbr_a, nm_a)
    cf_b, _ = predict_one(model, past_b, nbr_b_cf, nm_b)

    result = {
        "indices": pair,
        "track_ids": (int(windows.tids[i]), int(windows.tids[j])),
        "seq": int(windows.seqs[i]),
        "frame": int(windows.starts[i]),
        "intervention": (kind, value),
        "past_a": past_a, "past_b": past_b,
        "pred_a": pred_a, "pred_b": pred_b,
        "cf_pred_a": cf_a, "cf_pred_b": cf_b,
        "attention_a": attn_a, "attention_b": attn_b,
    }
    for key, (a, b) in (("original", (pred_a, pred_b)), ("counterfactual", (cf_a, cf_b))):
        scene = build_scene_graph(a, b)
        result[key] = {"scene": scene, "risk": scene["risk"], "explanation": explain(scene)}
    return result


def find_escalation(model, windows, kind="speed", value=2.0, max_pair_distance=15.0,
                    mask_neighbors=False):
    """Search all nearby co-occurring pairs for the intervention that raises risk the most.

    Returns the :func:`analyze_pair` result with the largest risk-tier increase
    (ties broken by the largest drop in closest approach), or ``None`` if the
    intervention never raises any pair's risk.
    """
    best, best_key = None, None
    for i, j in all_pairs(windows, max_pair_distance):
        res = analyze_pair(model, windows, kind, value, pair=(int(i), int(j)),
                           mask_neighbors=mask_neighbors)
        o, c = res["original"]["scene"], res["counterfactual"]["scene"]
        jump = RISK_ORDER.index(c["risk"]) - RISK_ORDER.index(o["risk"])
        if jump <= 0:
            continue
        key = (jump, o["min_distance_m"] - c["min_distance_m"])
        if best_key is None or key > best_key:
            best, best_key = res, key
    return best


def print_report(res):
    o, c = res["original"]["scene"], res["counterfactual"]["scene"]
    kind, value = res["intervention"]
    print(
        f"\nSequence {res['seq']:04d}, frame {res['frame']}, tracks {res['track_ids'][0]} (A) "
        f"and {res['track_ids'][1]} (B); intervention on A: {kind}={value}"
    )
    print("\n--- Metrics ---")
    print(f"Closest approach: {o['min_distance_m']:.2f} m  ->  {c['min_distance_m']:.2f} m")
    print(f"Closing speed:    {o['closing_speed_mps']:+.2f} m/s  ->  "
          f"{c['closing_speed_mps']:+.2f} m/s")
    print(f"TTC:              {o['ttc_s']:.2f} s  ->  {c['ttc_s']:.2f} s")
    print("\n--- Scene graph (original) ---")
    print(o)
    print("\n--- Scene graph (counterfactual) ---")
    print(c)
    print("\n--- Risk ---")
    print(f"Original:       {o['risk']}")
    print(f"Counterfactual: {c['risk']}")
    print("\n--- Explanation ---")
    print(res["original"]["explanation"])
    print(res["counterfactual"]["explanation"])


def figure_result(res):
    """Matplotlib figure comparing original and counterfactual predictions."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    for past, color in ((res["past_a"], "green"), (res["past_b"], "blue")):
        ax.plot(past[:, 0], past[:, 1], ":", color=color, alpha=0.6)
    ax.plot(*res["pred_a"].T, "o-", color="green", ms=3, label="A (predicted)")
    ax.plot(*res["pred_b"].T, "o-", color="blue", ms=3, label="B (predicted)")
    ax.plot(*res["cf_pred_a"].T, "o-", color="red", ms=3, label="A (counterfactual)")
    if not np.allclose(res["cf_pred_b"], res["pred_b"]):
        ax.plot(*res["cf_pred_b"].T, "o--", color="orange", ms=3, label="B (reacts)")
    step = max(1, len(res["pred_a"]) // 6)
    for k in range(0, len(res["pred_a"]), step):  # scale-free arrows A -> A_cf
        ax.annotate("", xy=res["cf_pred_a"][k], xytext=res["pred_a"][k],
                    arrowprops={"arrowstyle": "->", "color": "black", "lw": 0.8})
    kind, value = res["intervention"]
    ax.set_title(f"Counterfactual: {kind}={value} on A  "
                 f"({res['original']['risk']} -> {res['counterfactual']['risk']})")
    ax.set_xlabel("x (m, right)")
    ax.set_ylabel("z (m, forward)")
    ax.axis("equal")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return fig


def plot_result(res, save_path=None, show=True):
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = figure_result(res)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
