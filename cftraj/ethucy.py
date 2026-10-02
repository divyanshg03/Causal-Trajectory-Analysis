"""ETH/UCY pedestrian benchmark (leave-one-scene-out) as a second dataset.

Files use the common layout ``<root>/<scene>/{train,val,test}/*.txt`` with tab-separated
``frame  pedestrian_id  x  y`` in meters (world frame, 2.5 Hz, frames 10 apart) and the
standard protocol of 8 observed and 12 predicted steps. Every pedestrian scene has many
agents interacting in the open, unlike the sparse vehicle pairs of KITTI, so it is a better
test of whether the social model learns anything from its neighbors.
"""

import tempfile
from pathlib import Path

import numpy as np

from . import baselines
from .data import Windows, attach_neighbors, concat_windows, create_windows
from .metrics import ade_fde
from .model import load_model, predict_windows
from .train import train_model

SCENES = ("eth", "hotel", "univ", "zara1", "zara2")
ETHUCY_FPS = 2.5
PAST, FUTURE = 8, 12


def parse_ethucy_file(path):
    """``{pedestrian_id: [(step, x, y), ...]}`` with frames re-indexed to consecutive steps."""
    rows = []
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        rows.append((float(parts[0]), int(float(parts[1])), float(parts[2]), float(parts[3])))
    if not rows:
        return {}
    frames = np.array(sorted({r[0] for r in rows}))
    step = np.median(np.diff(frames)) if len(frames) > 1 else 1.0
    tracks = {}
    for frame, pid, x, y in rows:
        tracks.setdefault(pid, []).append((int(round((frame - frames[0]) / step)), x, y))
    return {pid: sorted(t) for pid, t in tracks.items()}


def load_scene_split(root, scene, split, past_len=PAST, future_len=FUTURE, seq0=0):
    """Windows (with neighbors) of every file in ``<root>/<scene>/<split>``."""
    files = sorted((Path(root) / scene / split).glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No {split} files under {Path(root) / scene / split}")
    parts = []
    for k, f in enumerate(files):
        tracks = {pid: t for pid, t in parse_ethucy_file(f).items()
                  if len(t) >= past_len + future_len}
        w = create_windows(tracks, past_len, future_len, seq=seq0 + k)
        if len(w):
            parts.append(attach_neighbors(w))
    if not parts:
        raise ValueError(f"No usable windows in {Path(root) / scene / split}")
    return concat_windows(parts)


def _score(name, pred, test, store):
    ade, fde = ade_fde(pred, test.Y)
    store.setdefault(name, {"ade": [], "fde": []})
    store[name]["ade"].append(ade)
    store[name]["fde"].append(fde)


def evaluate_scene(root, scene, seeds=(0,), epochs=60, device="cpu", archs=("transformer", "social")):
    """Baselines and ridge-prior models on one held-out scene; ``{method: {ade, fde}}``.

    For social models the test set is also scored with neighbors masked
    (``... (no context)``), which isolates what the model gets from other agents.
    """
    train = load_scene_split(root, scene, "train")
    val = load_scene_split(root, scene, "val", seq0=1000)
    test = load_scene_split(root, scene, "test", seq0=2000)
    res = {}
    q, r = baselines.tune_kalman(val.X, val.Y)
    _score("Stationary", baselines.stationary(test.X, FUTURE), test, res)
    _score("Constant velocity", baselines.constant_velocity(test.X, FUTURE), test, res)
    _score("Kalman", baselines.kalman(test.X, FUTURE, q, r), test, res)
    _score("Linear (ridge)", baselines.fit_ridge(train.X, train.Y)(test.X), test, res)
    for arch in archs:
        for seed in seeds:
            with tempfile.TemporaryDirectory() as tmp:
                ckpt = str(Path(tmp) / "m.pt")
                train_model(arch, train, val, ckpt, epochs=epochs, seed=seed, device=device,
                            log_every=999, linear_skip=True, residual=False)
                model = load_model(ckpt, device)
                name = f"{arch} + ridge prior"
                _score(name, predict_windows(model, test, device), test, res)
                if model.uses_neighbors:
                    _score(f"{name} (no context)",
                           predict_windows(model, test, device, mask_neighbors=True), test, res)
    res["_n"] = {"train": len(train), "val": len(val), "test": len(test)}
    return res


def run_ethucy(root, scenes=SCENES, **kw):
    """Leave-one-scene-out over ``scenes``; returns ``{scene: result}``."""
    out = {}
    for scene in scenes:
        print(f"== {scene} ==")
        out[scene] = evaluate_scene(root, scene, **kw)
    return out


def render(results):
    """Markdown: ADE / FDE per scene (mean over seeds) and the average across scenes."""
    scenes = list(results)
    methods = [m for m in results[scenes[0]] if m != "_n"]
    head = "| Method | " + " | ".join(scenes) + " | Average |"
    lines = ["### ETH/UCY (8 -> 12 steps at 2.5 Hz), ADE / FDE in meters", "", head,
             "|---|" + "---|" * (len(scenes) + 1)]
    for m in methods:
        cells, ades, fdes = [], [], []
        for s in scenes:
            ade, fde = np.mean(results[s][m]["ade"]), np.mean(results[s][m]["fde"])
            ades.append(ade)
            fdes.append(fde)
            cells.append(f"{ade:.2f} / {fde:.2f}")
        lines.append(f"| {m} | " + " | ".join(cells) + f" | {np.mean(ades):.2f} / "
                     f"{np.mean(fdes):.2f} |")
    sizes = ", ".join(f"{s}: {results[s]['_n']['test']:,}" for s in scenes)
    lines += ["", f"Test windows per scene: {sizes}."]
    return "\n".join(lines)


__all__ = ["SCENES", "Windows", "evaluate_scene", "load_scene_split", "parse_ethucy_file",
           "render", "run_ethucy"]
