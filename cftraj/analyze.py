"""End-to-end counterfactual collision analysis for one pair of agents."""

import numpy as np
import torch

from .counterfactual import scale_speed
from .pairs import find_interacting_pair
from .risk import build_scene_graph, closing_speed, explain, min_distance


@torch.no_grad()
def predict(model, past):
    x = torch.as_tensor(np.asarray(past), dtype=torch.float32).unsqueeze(0)
    return model(x).numpy()[0]


def analyze_pair(model, windows, factor=2.0, max_distance=None):
    """Compare predicted risk with and without a speed intervention on agent A.

    Returns ``None`` if the windows contain no co-occurring pair.
    """
    pair = find_interacting_pair(windows, max_distance)
    if pair is None:
        return None
    i, j = pair
    past1, past2 = windows.X[i], windows.X[j]

    pred1, pred2 = predict(model, past1), predict(model, past2)
    cf_pred1 = predict(model, scale_speed(past1, factor))

    result = {
        "indices": pair,
        "track_ids": (int(windows.tids[i]), int(windows.tids[j])),
        "frame": int(windows.starts[i]),
        "factor": factor,
        "pred_a": pred1,
        "pred_b": pred2,
        "cf_pred_a": cf_pred1,
    }
    for key, a in (("original", pred1), ("counterfactual", cf_pred1)):
        scene = build_scene_graph(a, pred2)
        result[key] = {
            "scene": scene,
            "min_distance": min_distance(a, pred2),
            "motion": closing_speed(a, pred2),
            "risk": scene["risk"],
            "explanation": explain(scene),
        }
    return result


def print_report(res):
    o, c = res["original"], res["counterfactual"]
    print(
        f"\nFrame {res['frame']}, tracks {res['track_ids'][0]} (A) and "
        f"{res['track_ids'][1]} (B); intervention: speed x{res['factor']} on A"
    )
    print("\n--- Metrics ---")
    print(f"Original min distance:       {o['min_distance']:.4f}")
    print(f"Counterfactual min distance: {c['min_distance']:.4f}")
    print(f"Closing speed (original):       {o['motion']:.6f}")
    print(f"Closing speed (counterfactual): {c['motion']:.6f}")
    print("\n--- Scene graph (original) ---")
    print(o["scene"])
    print("\n--- Scene graph (counterfactual) ---")
    print(c["scene"])
    print("\n--- Risk ---")
    print(f"Original:       {o['risk']}")
    print(f"Counterfactual: {c['risk']}")
    print("\n--- Explanation ---")
    print(o["explanation"])
    print(c["explanation"])


def plot_result(res, save_path=None, show=True):
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a, b, cf = res["pred_a"], res["pred_b"], res["cf_pred_a"]
    plt.figure(figsize=(6, 6))
    plt.plot(a[:, 0], a[:, 1], "go-", label="Object A (future)")
    plt.plot(b[:, 0], b[:, 1], "bo-", label="Object B (future)")
    plt.plot(cf[:, 0], cf[:, 1], "ro-", label="Object A (counterfactual)")
    for k in range(len(a)):
        plt.arrow(
            a[k, 0], a[k, 1], cf[k, 0] - a[k, 0], cf[k, 1] - a[k, 1],
            head_width=0.002, color="black",
        )
    plt.legend()
    plt.title("Counterfactual collision reasoning")
    plt.xlabel("x (normalized)")
    plt.ylabel("y (normalized)")
    plt.grid()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
