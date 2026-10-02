"""Benchmark learned models against classical baselines on the held-out test split."""

import json
from pathlib import Path

import numpy as np

from . import baselines
from .data import KITTI_FPS, load_split, split_label
from .metrics import ade_fde, bootstrap_ci, displacement_errors, error_by_horizon, min_ade_fde
from .model import load_model, predict_modes_numpy, predict_windows
from .risk import RISK_ORDER
from .train import pick_device
from .validity import (
    conflict_metrics,
    context_ablation,
    intervention_plausibility,
    neighbor_reaction,
    risk_calibration,
    speed_response,
)


def parse_ckpt_specs(specs):
    """``['transformer=a.pt,b.pt', 'lstm=c.pt']`` -> ``{'transformer': [...], ...}``."""
    out = {}
    for spec in specs:
        name, _, paths = spec.partition("=")
        if not paths:
            raise ValueError(f"Bad --ckpt '{spec}', expected NAME=path[,path...]")
        out[name] = paths.split(",")
    return out


def _fmt(vals, digits=3):
    vals = np.asarray(vals, dtype=float)
    s = f"{vals.mean():.{digits}f}"
    return s + (f" ± {vals.std():.{digits}f}" if len(vals) > 1 else "")


def _pct(vals):
    return _fmt(np.asarray(vals) * 100, 1)


def run_evaluation(label_dir, ckpts, out_dir="docs", device="auto", cache_dir=None):
    """Benchmark every checkpoint group; checkpoints of different horizons are evaluated
    in separate passes (each writes its own ``results_h<F>`` files)."""
    device = pick_device(device)
    models = {name: [load_model(p, device) for p in paths] for name, paths in ckpts.items()}
    by_len = {}
    for name, ms in models.items():
        lens = {(m.past_len, m.future_len) for m in ms}
        if len(lens) != 1:
            raise ValueError(f"Checkpoints of '{name}' mix horizons {sorted(lens)}; "
                             "list one model per horizon.")
        by_len.setdefault(lens.pop(), {})[name] = ms
    out = {}
    for (past_len, future_len), group in sorted(by_len.items(), key=lambda kv: kv[0][1]):
        group_ckpts = {name: ckpts[name] for name in group}
        out[future_len] = _evaluate_horizon(label_dir, group_ckpts, group, past_len, future_len,
                                            out_dir, device, cache_dir)
    return out if len(out) > 1 else next(iter(out.values()))


def _evaluate_horizon(label_dir, ckpts, models, past_len, future_len, out_dir, device,
                      cache_dir=None):
    train = load_split(label_dir, "train", past_len, future_len, cache_dir)
    val = load_split(label_dir, "val", past_len, future_len, cache_dir)
    test = load_split(label_dir, "test", past_len, future_len, cache_dir)
    print(f"windows: train {len(train):,}  val {len(val):,}  test {len(test):,}")

    q, r = baselines.tune_kalman(val.X, val.Y)
    ridge = baselines.fit_ridge(train.X, train.Y)
    print(f"Kalman tuned on val: q/r={q / r:g}")

    # name -> list of full test predictions (one per seed for learned models)
    preds = {
        "Stationary": [baselines.stationary(test.X, future_len)],
        "Constant velocity": [baselines.constant_velocity(test.X, future_len)],
        f"Kalman (q/r={q / r:g})": [baselines.kalman(test.X, future_len, q, r)],
        "Linear (ridge)": [ridge(test.X)],
    }
    for name, ms in models.items():
        preds[name] = [predict_windows(m, test, device) for m in ms]

    results = {}
    for name, plist in preds.items():
        ades, fdes, hz, conf = [], [], [], []
        for p in plist:
            ade, fde = ade_fde(p, test.Y)
            ades.append(ade)
            fdes.append(fde)
            hz.append(error_by_horizon(p, test.Y))
            conf.append(conflict_metrics(test, p))
        # 95% CI of the (seed-averaged) ADE, resampling whole tracks (windows of a track
        # are strongly correlated)
        per_window = np.mean([displacement_errors(p, test.Y).mean(1) for p in plist], axis=0)
        ci = bootstrap_ci(per_window, groups=test.seqs * 10**6 + test.tids)
        results[name] = {"ade": ades, "fde": fdes, "by_horizon": np.mean(hz, axis=0).tolist(),
                         "conflict": conf, "ade_ci95": list(ci[1:])}
    for name, ms in models.items():
        if ms[0].n_modes > 1:  # best-of-K metrics for multi-modal models
            mins = []
            for m in ms:
                futs, _ = predict_modes_numpy(m, test.X, device, nbr=test.nbr, nmask=test.nmask)
                mins.append(min_ade_fde(futs, test.Y))
            results[name]["min_ade"] = [v[0] for v in mins]
            results[name]["min_fde"] = [v[1] for v in mins]
            results[name]["n_modes"] = ms[0].n_modes

    validity = {}
    for name, ms in models.items():
        validity[name] = {
            "speed_response": [speed_response(m, test, device) for m in ms],
            "context": [context_ablation(m, test, device) for m in ms],
            "neighbor_reaction": [neighbor_reaction(m, test, device=device) for m in ms],
            "calibration": [risk_calibration(test, predict_windows(m, test, device))
                            for m in ms],
        }
    plaus = [intervention_plausibility(test, k, v)
             for k, v in (("speed", 2.0), ("speed", 0.5), ("brake", 0.8), ("lateral", 3.0))]

    text = _render(results, validity, test, past_len, future_len, plaus)
    print("\n" + text)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"h{future_len}"
    (out / f"results_{tag}.md").write_text(text + "\n", encoding="utf-8")
    (out / f"results_{tag}.json").write_text(
        json.dumps({"n_test": len(test), "kalman": {"q": q, "r": r}, "results": results,
                    "validity": validity, "plausibility": plaus}, indent=2),
        encoding="utf-8",
    )
    _plot_horizon(results, future_len, out / f"error_by_horizon_{tag}.png")
    _plot_curves(ckpts, out / f"training_curves_{tag}.png")
    print(f"\nWrote results_{tag}.md/.json and plots to {out}/")
    return results, validity


def _render(results, validity, test, past_len, future_len, plaus=()):
    horizon_s = future_len / KITTI_FPS
    n_pairs = results["Stationary"]["conflict"][0]["n_pairs"]
    n_pos = results["Stationary"]["conflict"][0]["n_positive"]
    lines = [
        (f"### Horizon {horizon_s:.1f} s ({past_len} past -> {future_len} future frames at "
         f"{KITTI_FPS:.0f} Hz)"),
        "",
        (f"Test split: KITTI tracking sequences {split_label('test')}, {len(test):,} windows. "
         "Errors in meters (bird's-eye view, ego-camera frame). Learned models: mean ± std "
         "over seeds."),
        "",
        "| Model | ADE (m) | ADE 95% CI | FDE (m) |",
        "|---|---|---|---|",
    ]
    for name, res in results.items():
        lo, hi = res["ade_ci95"]
        lines.append(f"| {name} | {_fmt(res['ade'])} | [{lo:.3f}, {hi:.3f}] | "
                     f"{_fmt(res['fde'])} |")
    multi = {n: r for n, r in results.items() if "min_ade" in r}
    if multi:
        lines += ["", "**Multi-modal models** (best of K modes per window):", "",
                  "| Model | K | minADE (m) | minFDE (m) |", "|---|---|---|---|"]
        for name, r in multi.items():
            lines.append(f"| {name} | {r['n_modes']} | {_fmt(r['min_ade'])} | "
                         f"{_fmt(r['min_fde'])} |")

    lines += [
        "",
        (f"**Conflict detection** — {n_pairs:,} not-yet-close co-occurring pairs (3-15 m apart "
         f"now), {n_pos:,} real conflicts (true future gap < 3 m). *Distance rule*: predicted "
         "gap < 3 m. *Risk rule*: MEDIUM or worse from TTC (see Risk logic)."),
        "",
        "| Model | Distance rule P / R / F1 (%) | Risk rule P / R / F1 (%) |",
        "|---|---|---|",
    ]
    for name, res in results.items():
        cells = []
        for rule in ("distance", "risk"):
            vals = [[c[rule][k] for c in res["conflict"] if rule in c]
                    for k in ("precision", "recall", "f1")]
            cells.append(" / ".join(_pct(v) for v in vals))
        lines.append(f"| {name} | {cells[0]} | {cells[1]} |")

    if validity:
        lines += [
            "",
            ("**Counterfactual validity** — predicted travel when an agent's past speed is "
             "scaled (constant velocity would give a ratio of 2.0 at 2x and 0 m at 0x)."),
            "",
            "| Model | Monotone in speed (%) | Travel ratio 2x / 1x | Travel at 0x (m) |",
            "|---|---|---|---|",
        ]
        for name, v in validity.items():
            sr = v["speed_response"]
            lines.append(
                f"| {name} | {_pct([s['monotone'] for s in sr])} | "
                f"{_fmt([s['travel_ratio_2x'] for s in sr], 2)} | "
                f"{_fmt([s['travel_at_zero_m'] for s in sr], 2)} |"
            )
        ctx = {n: v["context"] for n, v in validity.items() if v["context"][0] is not None}
        if ctx:
            lines += ["", "**Scene-context ablation** (social model, neighbors masked out):", "",
                      "| Model | ADE with context | ADE without | FDE with | FDE without |",
                      "|---|---|---|---|---|"]
            for name, c in ctx.items():
                lines.append(
                    f"| {name} | {_fmt([x['ade'] for x in c])} | "
                    f"{_fmt([x['ade_no_context'] for x in c])} | "
                    f"{_fmt([x['fde'] for x in c])} | {_fmt([x['fde_no_context'] for x in c])} |"
                )
        react = {n: v["neighbor_reaction"] for n, v in validity.items()
                 if v["neighbor_reaction"][0] is not None}
        if react:
            lines += ["", ("**Neighbor reaction** - shift of B's predicted final position (m) "
                           "when A's past speed is doubled; should shrink with distance:"), "",
                      "| Model | Mean | 95th pct | Pairs < 5 m | Pairs > 10 m |",
                      "|---|---|---|---|---|"]
            for name, rs in react.items():
                def col(key, rs=rs):
                    vals = [r[key] for r in rs if r.get(key) is not None]
                    return _fmt(vals, 3) if vals else "n/a"
                lines.append(f"| {name} | {col('mean_m')} | {col('p95_m')} | "
                             f"{col('mean_near_m')} | {col('mean_far_m')} |")
        lines += ["", ("**Risk-tier calibration** - fraction of pairs followed by a real "
                       "conflict (true future gap < 3 m), per predicted tier:"), "",
                  "| Model | " + " | ".join(RISK_ORDER) + " | F1 (MEDIUM+) | best TTC cut-off |",
                  "|---|" + "---|" * (len(RISK_ORDER) + 2)]
        for name, v in validity.items():
            cals = [c for c in v["calibration"] if "reliability" in c]
            if not cals:
                continue
            cells = []
            for tier in RISK_ORDER:
                rates = [c["reliability"][tier]["conflict_rate"] for c in cals
                         if c["reliability"][tier]["conflict_rate"] is not None]
                cells.append(_pct(rates) + "%" if rates else "n/a")
            lines.append(
                f"| {name} | " + " | ".join(cells) + " | "
                f"{_pct([c['default_f1'] for c in cals])}% | "
                f"{_fmt([c['best_ttc_threshold_s'] for c in cals], 1)} s |")
    if plaus:
        lines += ["", ("**Intervention plausibility** - share of real test histories that stay "
                       "under 10 m/s^2 after the intervention:"), "",
                  "| Intervention | Before | After |", "|---|---|---|"]
        for p in plaus:
            lines.append(f"| {p['kind']} = {p['value']} | {p['plausible_before'] * 100:.1f}% | "
                         f"{p['plausible_after'] * 100:.1f}% |")
    return "\n".join(lines)


def _plot_horizon(results, future_len, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.arange(1, future_len + 1) / KITTI_FPS
    plt.figure(figsize=(6, 4))
    for name, res in results.items():
        plt.plot(t, res["by_horizon"], marker="o", ms=3, label=name)
    plt.xlabel("prediction horizon (s)")
    plt.ylabel("mean displacement error (m)")
    plt.title("Test error vs horizon")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def _plot_curves(ckpts, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 4))
    drawn = False
    for ci, (name, paths) in enumerate(ckpts.items()):
        for k, p in enumerate(paths):
            hp = Path(f"{p}.history.json")
            if not hp.is_file():
                continue
            h = json.loads(hp.read_text())
            color = f"C{ci}"
            plt.plot(h["val_ade"], color=color, alpha=0.8, label=f"{name} val" if k == 0 else None)
            plt.plot(h["train_loss"], "--", color=color, alpha=0.5,
                     label=f"{name} train" if k == 0 else None)
            drawn = True
    if drawn:
        plt.xlabel("epoch")
        plt.ylabel("ADE (m)")
        plt.title("Training curves")
        plt.grid(alpha=0.3)
        plt.legend(fontsize=8)
        plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
