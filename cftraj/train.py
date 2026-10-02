import copy
import datetime
import json
import random
import subprocess

import numpy as np
import torch

from . import baselines
from .metrics import ade_fde, min_ade_fde
from .tracking import Tracker
from .model import build_model, predict_modes_numpy, predict_windows, save_checkpoint


def pick_device(name="auto"):
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


def run_info():
    """Provenance stored next to every checkpoint: package/torch versions, git commit, time."""
    from . import __version__

    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                                text=True, timeout=5, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        commit = None
    return {"cftraj": __version__, "torch": torch.__version__, "git_commit": commit,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ade_loss(pred, true):
    """Mean Euclidean displacement error (smooth at zero)."""
    return torch.sqrt(((pred - true) ** 2).sum(-1) + 1e-8).mean()


def validation_errors(model, w, device):
    """``(ADE, FDE)`` used for model selection: best-of-K for multi-modal models (their
    point of having K modes is coverage; the single top mode is a weak target), plain
    ADE/FDE otherwise."""
    if model.n_modes > 1:
        futs, _ = predict_modes_numpy(model, w.X, device, nbr=w.nbr, nmask=w.nmask)
        return min_ade_fde(futs, w.Y)
    return ade_fde(predict_windows(model, w, device), w.Y)


def multimodal_loss(modes, logits, true, cls_weight=0.1):
    """Winner-takes-all ADE over K modes plus cross-entropy on the winning mode's logit.

    ``modes`` is ``(B, K, F, 2)``, ``logits`` ``(B, K)``, ``true`` ``(B, F, 2)``.
    """
    err = torch.sqrt(((modes - true[:, None]) ** 2).sum(-1) + 1e-8).mean(-1)  # (B, K)
    best = err.argmin(dim=1)
    reg = err.gather(1, best[:, None]).mean()
    return reg + cls_weight * torch.nn.functional.cross_entropy(logits, best)


def train_model(arch, train_w, val_w, out_path, epochs=60, batch_size=256, lr=1e-3,
                weight_decay=1e-4, patience=10, seed=0, device="auto", log_every=1,
                augment=True, speed_aug=None, amp=False, run_args=None, tracker=None, **model_kwargs):
    """Mini-batch AdamW training with best-validation-ADE checkpointing.

    ``amp`` enables mixed precision on CUDA (ignored on CPU). ``run_args`` (e.g. the CLI
    arguments) is recorded with the run provenance. Multi-modal models
    (``n_modes > 1``) train with a winner-takes-all loss. Returns the history
    dict (also written to ``<out_path>.history.json``).
    """
    if len(train_w) == 0 or len(val_w) == 0:
        raise ValueError("Empty train or validation set.")
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    device = pick_device(device)
    set_seed(seed)

    config = {"past_len": train_w.past_len, "future_len": train_w.future_len, **model_kwargs}
    model = build_model(arch, **config).to(device)
    if model.skip is not None:  # start from the ridge solution; zero the learned correction
        model.init_skip(baselines.fit_ridge(train_w.X, train_w.Y).W)
        for layer in (model.fc,):
            if model.n_modes > 1:  # tiny random weights so the modes can specialise
                torch.nn.init.normal_(layer.weight, std=1e-2)
            else:
                torch.nn.init.zeros_(layer.weight)
            torch.nn.init.zeros_(layer.bias)
    tracker = tracker or Tracker()
    amp = bool(amp) and device.startswith("cuda")
    scaler = (torch.amp.GradScaler("cuda", enabled=amp) if hasattr(torch.amp, "GradScaler")
              else torch.cuda.amp.GradScaler(enabled=amp))  # torch < 2.3
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    X = torch.as_tensor(train_w.X, device=device)
    Y = torch.as_tensor(train_w.Y, device=device)
    social = model.uses_neighbors and train_w.nbr is not None
    NB = torch.as_tensor(train_w.nbr, device=device) if social else None
    NM = torch.as_tensor(train_w.nmask, device=device) if social else None
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{arch}] seed={seed} device={device} params={n_params:,} "
          f"horizon={train_w.future_len} train={len(X):,} val={len(val_w):,}")

    history = {"train_loss": [], "val_ade": [], "val_fde": [], "arch": arch, "seed": seed,
               "run": {**run_info(), "args": run_args, "amp": amp}}
    best_ade, best_state, best_epoch, bad = np.inf, None, 0, 0
    g = torch.Generator(device="cpu").manual_seed(seed)
    sign = torch.tensor([-1.0, 1.0], device=device)

    if model.skip is not None:  # the ridge-initialised model itself is a candidate (epoch 0)
        best_ade, best_fde = validation_errors(model, val_w, device)
        best_state, best_epoch = copy.deepcopy(model.state_dict()), 0
        print(f"  epoch   0  (ridge init)  val ADE {best_ade:.3f}  val FDE {best_fde:.3f}")

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(X), generator=g).to(device)
        total = 0.0
        for i in range(0, len(X), batch_size):
            idx = perm[i : i + batch_size]
            xb, yb = X[idx], Y[idx]
            nb = NB[idx] if social else None
            nm = NM[idx] if social else None
            if augment:  # mirror the scene left/right (x -> -x): physically valid
                flip = (torch.rand(len(idx), 1, 1, device=device) < 0.5).float()
                mirror = flip * sign + (1 - flip)
                xb, yb = xb * mirror, yb * mirror
                if social:
                    nb = nb * mirror[:, None]
            if speed_aug:  # re-play the target at a random speed about its current position
                s = torch.empty(len(idx), 1, 1, device=device).uniform_(*speed_aug)
                last = xb[:, -1:, :]
                xb, yb = last + s * (xb - last), last + s * (yb - last)
            opt.zero_grad()
            with torch.autocast(device_type="cuda", enabled=amp):
                if model.n_modes > 1:
                    modes, logits = model.forward_modes(xb, nb, nm)
                    loss = multimodal_loss(modes.float(), logits.float(), yb)
                else:
                    loss = ade_loss(model(xb, nb, nm).float(), yb)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            total += loss.item() * len(idx)
        sched.step()

        val_ade, val_fde = validation_errors(model, val_w, device)
        history["train_loss"].append(total / len(X))
        history["val_ade"].append(val_ade)
        history["val_fde"].append(val_fde)
        tracker.log({"train_loss": total / len(X), "val_ade": val_ade, "val_fde": val_fde},
                    step=epoch)
        if epoch % log_every == 0:
            print(f"  epoch {epoch:3d}  train ADE {total / len(X):.3f}  "
                  f"val ADE {val_ade:.3f}  val FDE {val_fde:.3f}")

        if val_ade < best_ade - 1e-4:
            best_ade, best_epoch, bad = val_ade, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= patience:
                print(f"  early stop at epoch {epoch} (best epoch {best_epoch})")
                break

    if best_state is None:
        raise RuntimeError(
            "Training never produced a finite validation ADE (diverged?). "
            "Try a smaller --lr."
        )
    model.load_state_dict(best_state)
    history.update(best_epoch=best_epoch, best_val_ade=best_ade)
    tracker.summary({"best_epoch": best_epoch, "best_val_ade": best_ade})
    save_checkpoint(out_path, model, arch, config, extra={"seed": seed})
    with open(f"{out_path}.history.json", "w") as f:
        json.dump(history, f)
    print(f"  saved {out_path} (best val ADE {best_ade:.3f} m @ epoch {best_epoch})")
    return history
