"""Tiny synthetic KITTI-format label files for smoke tests and CI (no dataset needed).

Agents move with a random constant velocity plus gentle curvature in the ground
plane, so they are learnable, cross paths, and co-occur in time. The files follow
the real label layout (``label_02/NNNN.txt``), so every code path that reads KITTI
labels, including the fixed train/val/test split, runs unchanged.
"""

from pathlib import Path

import numpy as np


def _line(frame, tid, x, z):
    # frame track type trunc occl alpha | bbox(4) | h w l | x y z | rot_y
    return f"{frame} {tid} Car 0 0 0 0 0 10 10 1.5 1.6 4.0 {x:.4f} 1.0 {z:.4f} 0.0\n"


def write_synthetic_kitti(label_dir, n_seqs=21, n_frames=60, n_agents=5, seed=0):
    """Write ``0000.txt ... `` into ``label_dir``; returns the directory path."""
    rng = np.random.default_rng(seed)
    out = Path(label_dir)
    out.mkdir(parents=True, exist_ok=True)
    t = np.arange(n_frames)
    for seq in range(n_seqs):
        lines = []
        for tid in range(n_agents):
            x0, z0 = rng.uniform(-12, 12), rng.uniform(8, 40)
            vx, vz = rng.uniform(-0.6, 0.6), rng.uniform(-0.8, 0.8)
            curve = rng.uniform(-0.004, 0.004)
            xs = x0 + vx * t + curve * t**2
            zs = z0 + vz * t
            lines += [_line(int(f), tid, x, z) for f, x, z in zip(t, xs, zs)]
        (out / f"{seq:04d}.txt").write_text("".join(lines))
    return out
