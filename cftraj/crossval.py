"""Sequence-level k-fold cross-validation.

The fixed 3-sequence test split is a small sample; this repeats the benchmark over
``k`` folds so each KITTI sequence is held out once. Baselines are cheap and always run;
a learned architecture is trained per fold when ``arch`` is given.
"""

import tempfile
from pathlib import Path

import numpy as np

from . import baselines
from .data import kfold_sequences, load_sequences
from .metrics import ade_fde
from .model import load_model, predict_windows
from .train import train_model


def cross_validate(label_dir, k=5, past_len=10, future_len=10, arch=None, epochs=30, seed=0,
                   device="cpu", cache_dir=None, ridge_prior=True):
    """Return ``{method: {"ade": [per fold], "fde": [per fold]}}`` plus fold bookkeeping."""
    folds = kfold_sequences(k)
    res = {}

    def add(name, pred, test):
        ade, fde = ade_fde(pred, test.Y)
        res.setdefault(name, {"ade": [], "fde": []})
        res[name]["ade"].append(ade)
        res[name]["fde"].append(fde)

    for n, (train_seqs, test_seqs) in enumerate(folds):
        val_seqs = train_seqs[::5]  # model selection / Kalman tuning, never the test fold
        fit_seqs = [s for s in train_seqs if s not in val_seqs]
        kw = {"past_len": past_len, "future_len": future_len, "cache_dir": cache_dir}
        train = load_sequences(label_dir, fit_seqs, **kw)
        val = load_sequences(label_dir, val_seqs, **kw)
        test = load_sequences(label_dir, test_seqs, **kw)
        print(f"fold {n + 1}/{k}: test sequences {test_seqs}  ({len(test):,} windows)")
        q, r = baselines.tune_kalman(val.X, val.Y)
        add("Stationary", baselines.stationary(test.X, future_len), test)
        add("Constant velocity", baselines.constant_velocity(test.X, future_len), test)
        add("Kalman", baselines.kalman(test.X, future_len, q, r), test)
        add("Linear (ridge)", baselines.fit_ridge(train.X, train.Y)(test.X), test)
        if arch:
            with tempfile.TemporaryDirectory() as tmp:
                ckpt = str(Path(tmp) / "m.pt")
                extra = {"linear_skip": True, "residual": False} if ridge_prior else {}
                train_model(arch, train, val, ckpt, epochs=epochs, seed=seed, device=device,
                            **extra)
                model = load_model(ckpt, device)
                add(f"{arch}{' + ridge' if ridge_prior else ''}",
                    predict_windows(model, test, device), test)
    return {"k": k, "folds": [t for _, t in folds], "results": res}


def render(cv):
    """Markdown table: mean ± std over folds."""
    lines = [f"### {cv['k']}-fold cross-validation by sequence", "",
             "| Method | ADE (m) | FDE (m) |", "|---|---|---|"]
    for name, r in cv["results"].items():
        ade, fde = np.asarray(r["ade"]), np.asarray(r["fde"])
        lines.append(f"| {name} | {ade.mean():.3f} ± {ade.std():.3f} | "
                     f"{fde.mean():.3f} ± {fde.std():.3f} |")
    return "\n".join(lines)
