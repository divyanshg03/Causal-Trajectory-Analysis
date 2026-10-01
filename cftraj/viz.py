"""Visualizations: attention over neighbors and a camera + bird's-eye-view demo GIF."""

from pathlib import Path

import numpy as np

from .analyze import predict_one
from .data import group_indices, parse_kitti_labels
from .model import predict_numpy
from .pairs import all_pairs
from .risk import RISK_ORDER, assess_pair
from .validity import CONFLICT_RADIUS

RISK_COLORS = {
    "SAFE": "#9aa5b1",
    "LOW": "#7ac07a",
    "MEDIUM": "#f2b134",
    "HIGH": "#ee7b30",
    "CRITICAL": "#d62728",
}


def plot_attention(model, windows, i, save_path=None, show=False):
    """Draw agent ``i``'s scene; neighbor line width is its attention weight."""
    import matplotlib

    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not model.uses_neighbors:
        raise ValueError("Attention plots need a social model (--arch social).")
    pred, attn = predict_one(model, windows.X[i], windows.nbr[i], windows.nmask[i])

    fig, ax = plt.subplots(figsize=(6, 6))
    same = np.flatnonzero(windows.group_keys() == windows.group_keys()[i])
    for k in same:
        ax.plot(*windows.X[k].T, "-", color="#c3c9d0", lw=1)
        ax.plot(*windows.X[k][-1], "o", color="#9aa5b1", ms=4)
    target = windows.X[i][-1]
    ax.plot(*windows.X[i].T, "-", color="#d62728", lw=2)
    ax.plot(*pred.T, ".-", color="#d62728", ms=3, label="target prediction")
    for slot in np.flatnonzero(windows.nmask[i]):
        pos = windows.nbr[i, slot, -1]
        ax.plot([target[0], pos[0]], [target[1], pos[1]], color="#1f77b4",
                lw=0.5 + 8 * float(attn[slot]), alpha=0.8)
        ax.plot(*pos, "o", color="#1f77b4", ms=6)
        ax.annotate(f"{attn[slot]:.2f}", pos, textcoords="offset points", xytext=(5, 5),
                    fontsize=8, color="#1f77b4")
    ax.plot(*target, "o", color="#d62728", ms=9, label="target")
    ax.set_title("Attention over neighbors (line width ∝ weight)")
    ax.set_xlabel("x (m, right)")
    ax.set_ylabel("z (m, forward)")
    ax.axis("equal")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return attn


def scene_risk(windows, preds, idx, pair_radius=15.0):
    """Worst risk per agent in a frame group. Returns ({agent index: (risk, ttc)}, pairs)."""
    sub_pairs = all_pairs(_Subset(windows, idx), pair_radius)
    worst = {int(a): ("SAFE", np.inf) for a in range(len(idx))}
    for a, b in sub_pairs:
        if np.linalg.norm(windows.X[idx[a], -1] - windows.X[idx[b], -1]) < CONFLICT_RADIUS:
            continue  # already together (e.g. pedestrians walking side by side): not a new risk
        scene = assess_pair(preds[a], preds[b])
        for k in (int(a), int(b)):
            if RISK_ORDER.index(scene["risk"]) > RISK_ORDER.index(worst[k][0]):
                worst[k] = (scene["risk"], scene["ttc_s"])
    return worst, sub_pairs


class _Subset:
    """Minimal window view with the attributes ``all_pairs`` needs."""

    def __init__(self, w, idx):
        self.X, self.seqs, self.starts = w.X[idx], w.seqs[idx], w.starts[idx]

    def group_keys(self):
        return self.seqs.astype(np.int64) * 10**7 + self.starts


def make_demo(model, windows, label_path, image_dir, out_path, first_frame=0, n_frames=60,
              step=2, fps=5, device="cpu"):
    """Render a camera + bird's-eye-view GIF colored by predicted pairwise risk."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    from PIL import Image

    labels = parse_kitti_labels(label_path)
    groups = {int(windows.starts[g[0]]): g for g in group_indices(windows.group_keys())}
    now_offset = windows.past_len - 1  # a window starting at s is "now" at s + P - 1
    frames_out = []
    for t in range(first_frame, first_frame + n_frames * step, step):
        g = groups.get(t - now_offset)
        img_path = Path(image_dir) / f"{t:06d}.png"
        if g is None or not img_path.is_file():
            continue
        preds = predict_numpy(
            model, windows.X[g], device,
            nbr=None if windows.nbr is None else windows.nbr[g],
            nmask=None if windows.nmask is None else windows.nmask[g],
        )
        worst, _ = scene_risk(windows, preds, g)

        fig, (ax_img, ax_bev) = plt.subplots(1, 2, figsize=(11, 3.4), dpi=72,
                                             gridspec_kw={"width_ratios": [2.6, 1]})
        ax_img.imshow(mpimg.imread(img_path))
        ax_img.axis("off")
        by_tid = {o["track_id"]: o for o in labels.get(t, [])}
        for k, wi in enumerate(g):
            risk, ttc = worst[k]
            color = RISK_COLORS[risk]
            obj = by_tid.get(int(windows.tids[wi]))
            if obj is not None:
                x1, y1, x2, y2 = obj["bbox"]
                ax_img.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                               ec=color, lw=2.5 if risk != "SAFE" else 1.2))
                if risk != "SAFE":
                    ax_img.text(x1, y1 - 3, f"{risk} {ttc:.1f}s" if np.isfinite(ttc) else risk,
                                color="white", fontsize=7,
                                bbox={"fc": color, "ec": "none", "pad": 1})
            ax_bev.plot(*windows.X[wi].T, "-", color=color, alpha=0.5, lw=1)
            ax_bev.plot(*preds[k].T, "-", color=color, lw=1.8)
            ax_bev.plot(*windows.Y[wi].T, ":", color="black", alpha=0.35, lw=1)
            ax_bev.plot(*windows.X[wi][-1], "o", color=color, ms=5)
        ax_bev.set_title("BEV: past, predicted (solid), true future (dotted)", fontsize=8)
        ax_bev.set_xlabel("x (m)")
        ax_bev.set_ylabel("z (m)")
        ax_bev.set_xlim(-25, 25)
        ax_bev.set_ylim(0, 60)
        ax_bev.grid(alpha=0.3)
        fig.suptitle(f"Seq {int(windows.seqs[g[0]]):04d}  frame {t}  "
                     "(box color = worst predicted pair risk)", fontsize=9)
        fig.tight_layout()
        fig.canvas.draw()
        frames_out.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]))
        plt.close(fig)

    if not frames_out:
        raise ValueError("No frames rendered: check --first-frame and the image directory.")
    frames_out[0].save(out_path, save_all=True, append_images=frames_out[1:],
                       duration=int(1000 / fps), loop=0, optimize=True)
    return len(frames_out)
