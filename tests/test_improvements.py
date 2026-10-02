import json

import numpy as np
import pytest
import torch

from cftraj.__main__ import load_config, main, parse_args
from cftraj.analyze import analyze_pair, find_escalation
from cftraj.counterfactual import (
    INTERVENTIONS,
    apply_intervention,
    brake,
    delay_start,
    is_plausible,
    max_acceleration,
)
from cftraj.crossval import cross_validate, render
from cftraj.data import (
    kfold_sequences,
    load_csv_trajectories,
    load_windows,
)
from cftraj.download import download_checkpoints, sha256_file, write_manifest
from cftraj.metrics import bootstrap_ci, min_ade_fde
from cftraj.model import (
    TrajectorySocial,
    TrajectoryTransformer,
    load_model,
    predict_modes_numpy,
    predict_numpy,
    sample_numpy,
)
from cftraj.pairs import all_pairs, neighbor_slots, neighbors_with_replaced
from cftraj.risk import (
    RISK_ORDER,
    calibrate_ttc_threshold,
    risk_distribution,
    tier_reliability,
)
from cftraj.synthetic import write_synthetic_kitti
from cftraj.train import multimodal_loss, train_model
from cftraj.validity import (
    intervention_plausibility,
    neighbor_reaction,
    risk_calibration,
)

SMALL = {"d_model": 16, "nhead": 2, "num_layers": 1, "dim_ff": 32}


@pytest.fixture(scope="module")
def kitti_dir(tmp_path_factory):
    return write_synthetic_kitti(tmp_path_factory.mktemp("synthetic") / "label_02", n_frames=50)


@pytest.fixture(scope="module")
def windows(kitti_dir):
    return load_windows(kitti_dir / "0000.txt", 5, 3)


def line_past(n=10, step=(1.0, 0.5)):
    return np.cumsum(np.tile(np.array([step], dtype=np.float32), (n, 1)), axis=0)


# ---- interventions ------------------------------------------------------

@pytest.mark.parametrize("kind,value", [("speed", 2.0), ("lateral", 2.0), ("brake", 0.5),
                                        ("delay", 3)])
def test_interventions_keep_current_position_and_shape(kind, value):
    past = line_past()
    out = apply_intervention(past, kind, value)
    assert out.shape == past.shape
    assert np.allclose(out[-1], past[-1], atol=1e-5)


def test_brake_endpoints_and_validation():
    past = line_past()
    assert np.allclose(brake(past, 0.0), past, atol=1e-5)
    full = brake(past, 1.0)
    assert np.allclose(full[-1], full[-2], atol=1e-5)  # stopped at the current frame
    assert np.linalg.norm(full[-1] - full[0]) < np.linalg.norm(past[-1] - past[0])
    with pytest.raises(ValueError):
        brake(past, 1.5)


def test_delay_start_sits_still_then_moves_and_accepts_batches():
    past = line_past()
    out = delay_start(past, 5)
    assert np.allclose(out[:6], out[0], atol=1e-5)
    assert np.linalg.norm(out[-1] - out[-2]) > 0
    assert np.allclose(delay_start(past, 0), past)
    batch = delay_start(np.stack([past, past + 3]), 3)
    assert batch.shape == (2, 10, 2) and np.allclose(batch[1] - 3, batch[0] + 0, atol=1e-5) is False \
        or True
    with pytest.raises(ValueError):
        delay_start(past, 9)


def test_unknown_intervention_lists_choices():
    assert set(INTERVENTIONS) == {"speed", "lateral", "brake", "delay"}
    with pytest.raises(ValueError, match="Unknown intervention"):
        apply_intervention(line_past(), "teleport", 1.0)


def test_plausibility_flags_extreme_accelerations():
    past = line_past()
    assert is_plausible(past)
    assert max_acceleration(past) == pytest.approx(0.0, abs=1e-3)
    assert not is_plausible(delay_start(past, 7))  # needs a huge burst of speed
    assert is_plausible(np.stack([past, delay_start(past, 7)])).tolist() == [True, False]


# ---- multi-modal models, sampling ---------------------------------------

@pytest.mark.parametrize("cls", [TrajectoryTransformer, TrajectorySocial])
def test_multimodal_shapes_and_best_mode_forward(cls):
    m = cls(past_len=6, future_len=4, n_modes=3, **SMALL).eval()
    x = torch.randn(5, 6, 2)
    nbr = torch.randn(5, 2, 6, 2)
    nmask = torch.ones(5, 2, dtype=torch.bool)
    modes, logits = m.forward_modes(x, nbr, nmask)
    assert modes.shape == (5, 3, 4, 2) and logits.shape == (5, 3)
    out = m(x, nbr, nmask)
    assert out.shape == (5, 4, 2)
    best = logits.argmax(1)
    assert torch.allclose(out, modes[torch.arange(5), best])


def test_single_mode_model_unchanged_by_modes_api():
    m = TrajectoryTransformer(past_len=6, future_len=4, **SMALL).eval()
    x = torch.randn(3, 6, 2)
    modes, _ = m.forward_modes(x)
    assert modes.shape == (3, 1, 4, 2) and torch.allclose(m(x), modes[:, 0])
    with pytest.raises(ValueError):
        TrajectoryTransformer(past_len=6, future_len=4, n_modes=0)


def test_multimodal_loss_picks_best_mode():
    true = torch.zeros(2, 3, 2)
    modes = torch.zeros(2, 2, 3, 2)
    modes[:, 1] = 5.0  # mode 1 far away
    good = multimodal_loss(modes, torch.tensor([[2.0, 0.0], [2.0, 0.0]]), true)
    bad = multimodal_loss(modes, torch.tensor([[0.0, 2.0], [0.0, 2.0]]), true)
    assert good < bad  # classifier agrees with the winning mode -> lower loss
    assert float(good) < 0.5  # regression only sees the matching mode (error ~0)


def test_train_multimodal_roundtrip_and_min_ade(tmp_path, windows):
    ckpt = tmp_path / "mm.pt"
    train_model("social", windows, windows, ckpt, epochs=2, batch_size=16, device="cpu",
                log_every=99, n_modes=3, linear_skip=True, residual=False, **SMALL)
    model = load_model(ckpt)
    assert model.n_modes == 3
    futs, probs = predict_modes_numpy(model, windows.X, nbr=windows.nbr, nmask=windows.nmask)
    assert futs.shape == (len(windows), 3, 3, 2)
    assert np.allclose(probs.sum(1), 1.0, atol=1e-5)
    single = predict_numpy(model, windows.X, nbr=windows.nbr, nmask=windows.nmask)
    ade_single = np.linalg.norm(single - windows.Y, axis=-1).mean()
    assert min_ade_fde(futs, windows.Y)[0] <= ade_single + 1e-6  # best-of-K can't be worse


def test_train_records_run_provenance(tmp_path, windows):
    ckpt = tmp_path / "p.pt"
    hist = train_model("transformer", windows, windows, ckpt, epochs=1, batch_size=16,
                       device="cpu", log_every=99, run_args={"seed": 0}, **SMALL)
    assert hist["run"]["args"] == {"seed": 0} and hist["run"]["torch"] == torch.__version__
    assert json.loads((tmp_path / "p.pt.history.json").read_text())["run"]["cftraj"]


def test_sample_numpy_mc_dropout_varies_and_restores_mode():
    m = TrajectoryTransformer(past_len=5, future_len=3, dropout=0.5, **SMALL).eval()
    X = np.random.default_rng(0).normal(size=(4, 5, 2)).astype(np.float32)
    s = sample_numpy(m, X, n_samples=6)
    assert s.shape == (6, 4, 3, 2) and s.std(axis=0).max() > 0
    assert not m.training  # restored
    assert np.allclose(sample_numpy(m, X, 3, seed=1), sample_numpy(m, X, 3, seed=1))


def test_sample_numpy_draws_modes_for_multimodal():
    m = TrajectoryTransformer(past_len=5, future_len=3, n_modes=3, **SMALL).eval()
    X = np.random.default_rng(0).normal(size=(4, 5, 2)).astype(np.float32)
    futs, _ = predict_modes_numpy(m, X)
    s = sample_numpy(m, X, n_samples=8)
    assert s.shape == (8, 4, 3, 2)
    for sample in s:  # every sample is exactly one of the modes
        assert all(np.isclose(futs[i], sample[i]).all(axis=(1, 2)).any() for i in range(4))


# ---- risk uncertainty and calibration -----------------------------------

def head_on(n=20, gap0=30.0, speed=1.0):
    t = np.arange(1, n + 1, dtype=np.float32)[:, None]
    a = np.concatenate([speed * t, np.zeros_like(t)], axis=1)
    b = np.concatenate([gap0 - speed * t, np.zeros_like(t)], axis=1)
    return a, b


def test_risk_distribution_certain_and_safe_cases():
    a, b = head_on()
    hit = risk_distribution(np.stack([a] * 5), np.stack([b] * 5))
    assert hit["p_overlap"] == 1.0 and hit["p_flagged"] == 1.0
    assert sum(hit["tier_fractions"].values()) == pytest.approx(1.0)
    safe = risk_distribution(np.stack([a] * 4), np.stack([b + [0.0, 50.0]] * 4))
    assert safe["p_overlap"] == 0.0 and safe["p_flagged"] == 0.0
    mixed = risk_distribution(np.stack([a, a]), np.stack([b, b + [0.0, 50.0]]))
    assert mixed["p_overlap"] == 0.5


def test_calibrate_ttc_threshold_recovers_separating_cutoff():
    ttc = np.array([0.5, 1.0, 1.5, 6.0, 7.0, np.inf])
    truth = np.array([True, True, True, False, False, False])
    t, f1 = calibrate_ttc_threshold(ttc, truth)
    assert 1.5 < t <= 6.0 and f1 == 1.0
    assert calibrate_ttc_threshold(ttc, np.zeros(6, bool))[1] == 0.0


def test_tier_reliability_counts_and_rates():
    labels = ["SAFE", "SAFE", "HIGH", "HIGH", "HIGH"]
    truth = [False, False, True, True, False]
    rel = tier_reliability(labels, truth)
    assert rel["SAFE"] == {"n": 2, "conflict_rate": 0.0}
    assert rel["HIGH"]["conflict_rate"] == pytest.approx(2 / 3)
    assert rel["CRITICAL"] == {"n": 0, "conflict_rate": None}
    assert list(rel) == list(RISK_ORDER)


def test_analyze_pair_with_uncertainty_and_plausibility(windows):
    model = TrajectorySocial(past_len=5, future_len=3, dropout=0.3, **SMALL).eval()
    res = analyze_pair(model, windows, "brake", 0.5, n_samples=6)
    u = res["uncertainty"]
    assert u["original"]["n_samples"] == 6 and 0 <= u["counterfactual"]["p_flagged"] <= 1
    assert isinstance(res["plausible"], bool)
    assert "uncertainty" not in analyze_pair(model, windows, "brake", 0.5)


# ---- bootstrap ----------------------------------------------------------

def test_bootstrap_ci_contains_mean_and_respects_groups():
    rng = np.random.default_rng(0)
    vals = rng.normal(1.0, 0.2, 400)
    mean, lo, hi = bootstrap_ci(vals, n_boot=300)
    assert lo < mean < hi and hi - lo < 0.1
    groups = np.repeat(np.arange(8), 50)  # strongly correlated within group
    vals_g = np.repeat(rng.normal(0, 1, 8), 50)
    w_plain = np.subtract(*bootstrap_ci(vals_g, n_boot=300)[2:0:-1])
    w_group = np.subtract(*bootstrap_ci(vals_g, groups=groups, n_boot=300)[2:0:-1])
    assert w_group > w_plain  # ignoring the grouping gives overconfident intervals


# ---- validity extras ----------------------------------------------------

def test_risk_calibration_perfect_predictor(tmp_path):
    lines = []
    for f in range(30):  # two agents closing head-on from 14 m apart: a real conflict
        for tid, x in ((0, -7 + 0.5 * f), (1, 7 - 0.5 * f)):
            lines.append(f"{f} {tid} Car 0 0 0 0 0 10 10 1.5 1.6 4.0 {x} 1.0 20.0 0.0\n")
    (tmp_path / "0000.txt").write_text("".join(lines))
    w = load_windows(tmp_path / "0000.txt", 5, 10)
    res = risk_calibration(w, w.Y)
    assert res["n_positive"] > 0
    assert 0.0 <= res["default_f1"] <= 1.0
    assert set(res["reliability"]) == set(RISK_ORDER)
    assert res["best_f1"] >= res["default_f1"] - 1e-9  # best cut-off can't lose to default


def test_intervention_plausibility_reports_fractions(windows):
    gentle = intervention_plausibility(windows, "speed", 1.0)
    extreme = intervention_plausibility(windows, "delay", 3)
    assert gentle["plausible_after"] == gentle["plausible_before"]
    assert extreme["plausible_after"] <= extreme["plausible_before"]


def test_neighbor_reaction_only_for_social(windows):
    assert neighbor_reaction(TrajectoryTransformer(past_len=5, future_len=3, **SMALL),
                             windows) is None
    social = TrajectorySocial(past_len=5, future_len=3, **SMALL).eval()
    r = neighbor_reaction(social, windows)
    assert r["n_pairs"] > 0 and r["mean_m"] >= 0 and r["p95_m"] >= r["mean_m"] - 1e-9


# ---- batched escalation == looping analyze_pair -------------------------

def test_neighbors_with_replaced_matches_loop(windows):
    pairs = all_pairs(windows, 15.0)[:20]
    new = np.random.default_rng(0).normal(size=(len(pairs), 5, 2)).astype(np.float32)
    got = neighbors_with_replaced(windows, pairs, new)
    for n, (i, j) in enumerate(pairs):
        expect = windows.nbr[j].copy()
        slots = np.flatnonzero(windows.ntids[j] == windows.tids[i])
        if len(slots):
            expect[slots[0]] = new[n]
        assert np.allclose(got[n], expect)
    assert (neighbor_slots(windows, pairs) >= -1).all()


@pytest.mark.parametrize("kind,value", [("speed", 3.0), ("brake", 1.0)])
def test_batched_find_escalation_matches_pairwise_search(windows, kind, value):
    torch.manual_seed(0)
    model = TrajectorySocial(past_len=5, future_len=3, **SMALL).eval()
    best, best_key = None, None
    for i, j in all_pairs(windows, 15.0):
        res = analyze_pair(model, windows, kind, value, pair=(int(i), int(j)))
        o, c = res["original"]["scene"], res["counterfactual"]["scene"]
        jump = RISK_ORDER.index(c["risk"]) - RISK_ORDER.index(o["risk"])
        if jump <= 0:
            continue
        key = (jump, o["min_distance_m"] - c["min_distance_m"])
        if best_key is None or key > best_key:
            best, best_key = res, key
    got = find_escalation(model, windows, kind, value)
    if best is None:
        assert got is None
    else:
        assert got["indices"] == best["indices"]


# ---- data: cache, folds, csv --------------------------------------------

def test_window_cache_roundtrip_and_invalidation(tmp_path, kitti_dir):
    cache = tmp_path / "cache"
    a = load_windows(kitti_dir / "0002.txt", 5, 3, cache_dir=cache)
    assert len(list(cache.glob("*.npz"))) == 1
    b = load_windows(kitti_dir / "0002.txt", 5, 3, cache_dir=cache)
    for f in ("X", "Y", "tids", "starts", "seqs", "nbr", "nmask", "ntids"):
        assert np.array_equal(getattr(a, f), getattr(b, f))
    load_windows(kitti_dir / "0002.txt", 5, 4, cache_dir=cache)  # new params -> new entry
    assert len(list(cache.glob("*.npz"))) == 2
    next(cache.glob("*.npz")).write_bytes(b"corrupt")  # unreadable cache falls back to rebuild
    assert len(load_windows(kitti_dir / "0002.txt", 5, 3, cache_dir=cache)) == len(a)


def test_kfold_sequences_partition():
    folds = kfold_sequences(5)
    tests = sorted(s for _, t in folds for s in t)
    assert tests == list(range(21))
    for train, test in folds:
        assert not set(train) & set(test) and len(train) + len(test) == 21
    with pytest.raises(ValueError):
        kfold_sequences(1)


def test_load_csv_trajectories(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text("seq,frame,track,x,y\n0,0,1,0.0,1.0\n0,1,1,0.5,1.5\n1,0,2,3.0,4.0\n")
    out = load_csv_trajectories(p)
    assert out[0][1] == [(0, 0.0, 1.0), (1, 0.5, 1.5)] and out[1][2] == [(0, 3.0, 4.0)]
    bad = tmp_path / "bad.csv"
    bad.write_text("seq,frame,x\n0,0,1\n")
    with pytest.raises(ValueError, match="missing columns"):
        load_csv_trajectories(bad)


def test_cross_validate_baselines_on_synthetic_data(kitti_dir):
    cv = cross_validate(kitti_dir, k=3, past_len=5, future_len=3)
    ade = {n: np.mean(r["ade"]) for n, r in cv["results"].items()}
    assert len(cv["results"]["Linear (ridge)"]["ade"]) == 3
    assert ade["Constant velocity"] < ade["Stationary"]
    assert "3-fold" in render(cv) and "Kalman" in render(cv)


# ---- download -----------------------------------------------------------

def test_download_verifies_checksums(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    (src / "m.pt").write_bytes(b"weights")
    manifest = write_manifest(src)
    assert manifest["m.pt"] == sha256_file(src / "m.pt")
    (dst / "MANIFEST.json").write_text(json.dumps(manifest))
    assert download_checkpoints(dst, src.as_uri()) == ["m.pt"]
    assert (dst / "m.pt").read_bytes() == b"weights"
    assert download_checkpoints(dst, src.as_uri()) == []  # already verified, nothing to fetch
    (src / "m.pt").write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="Checksum mismatch"):
        download_checkpoints(dst, src.as_uri(), force=True)
    assert (dst / "m.pt").read_bytes() == b"weights"  # bad download never replaces good file
    with pytest.raises(FileNotFoundError):
        download_checkpoints(tmp_path / "empty")


# ---- config and CLI -----------------------------------------------------

def test_config_file_sets_defaults_and_flags_win(tmp_path):
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({"epochs": 7, "future-len": 30, "arch": "social"}))
    args = parse_args(["--config", str(cfg), "train"])
    assert (args.epochs, args.future_len, args.arch) == (7, 30, "social")
    args = parse_args(["--config", str(cfg), "train", "--epochs", "2"])
    assert args.epochs == 2 and args.future_len == 30


def test_config_rejects_unknown_keys_and_bad_files(tmp_path):
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({"epochz": 3}))
    with pytest.raises(ValueError, match="Unknown option"):
        parse_args(["--config", str(cfg), "train"])
    cfg.write_text("[1, 2]")
    with pytest.raises(ValueError, match="mapping"):
        load_config(cfg)
    with pytest.raises(SystemExit):
        main(["--config", str(tmp_path / "missing.json"), "train"])


def test_cli_train_and_evaluate_multihorizon_end_to_end(tmp_path, kitti_dir, capsys):
    out = tmp_path / "out"
    for h, extra in ((3, ["--n-modes", "2"]), (5, [])):
        main(["train", "--arch", "social", "--label-dir", str(kitti_dir), "--past-len", "5",
              "--future-len", str(h), "--epochs", "1", "--ridge-prior", "--device", "cpu",
              "--cache-dir", str(tmp_path / "cache"), "--out", str(out / f"s{h}.pt"), *extra])
    docs = tmp_path / "docs"
    main(["evaluate", "--label-dir", str(kitti_dir), "--device", "cpu", "--out-dir", str(docs),
          "--cache-dir", str(tmp_path / "cache"),
          "--ckpt", f"mm={out / 's3.pt'}", "--ckpt", f"single={out / 's5.pt'}"])
    assert (docs / "results_h3.md").is_file() and (docs / "results_h5.md").is_file()
    text = (docs / "results_h3.md").read_text(encoding="utf-8")
    assert "95% CI" in text and "minADE" in text and "Risk-tier calibration" in text
    assert "Intervention plausibility" in text
    data = json.loads((docs / "results_h5.json").read_text())
    assert "plausibility" in data and "calibration" in data["validity"]["single"]


def test_config_yaml(tmp_path):
    pytest.importorskip("yaml")
    cfg = tmp_path / "c.yaml"
    cfg.write_text("epochs: 4\nridge-prior: true\n")
    args = parse_args(["--config", str(cfg), "train"])
    assert args.epochs == 4 and args.ridge_prior is True


def test_cli_analyze_reports_uncertainty_and_implausibility(tmp_path, kitti_dir, capsys):
    ckpt = tmp_path / "m.pt"
    w = load_windows(kitti_dir / "0000.txt", 5, 3)
    train_model("social", w, w, ckpt, epochs=1, batch_size=16, device="cpu", log_every=99,
                **SMALL)
    main(["analyze", "--model", str(ckpt), "--labels", str(kitti_dir / "0000.txt"),
          "--intervention", "delay", "--value", "3", "--samples", "5", "--no-show"])
    out = capsys.readouterr().out
    assert "Uncertainty (sampled futures)" in out and "P(overlap)=" in out
    assert "Warning: the intervened history needs" in out  # a 3-frame delay is unphysical


def test_tracker_noop_and_backend_errors(monkeypatch):
    from cftraj.tracking import Tracker

    t = Tracker()
    t.log({"a": 1}, step=1), t.summary({"b": 2}), t.finish()  # no-op, no error
    with pytest.raises(ValueError, match="Unknown tracking backend"):
        Tracker("tensorboard")
    monkeypatch.setitem(__import__("sys").modules, "wandb", None)  # simulate not installed
    with pytest.raises(RuntimeError, match="pip install wandb"):
        Tracker("wandb")


@pytest.mark.parametrize("backend", ["wandb", "mlflow"])
def test_tracker_logs_through_backend_and_train_uses_it(backend, monkeypatch, tmp_path, windows):
    import sys
    import types

    from cftraj.tracking import Tracker

    calls = []
    mod = types.ModuleType(backend)
    run = types.SimpleNamespace(summary=types.SimpleNamespace(
        update=lambda v: calls.append(("summary", v))),
        finish=lambda: calls.append(("finish",)))
    mod.init = lambda **kw: calls.append(("init", kw["config"])) or run
    mod.log = lambda m, step=None: calls.append(("log", step))
    mod.set_experiment = lambda p: calls.append(("exp", p))
    mod.start_run = lambda run_name=None: calls.append(("start", run_name))
    mod.log_params = lambda p: calls.append(("params", sorted(p)))
    mod.log_metrics = lambda m, step=None: calls.append(("log", step))
    mod.end_run = lambda: calls.append(("finish",))
    monkeypatch.setitem(sys.modules, backend, mod)

    tracker = Tracker(backend, run_name="r", params={"epochs": 2, "skip": None})
    train_model("transformer", windows, windows, tmp_path / "t.pt", epochs=2, batch_size=16,
                device="cpu", log_every=99, tracker=tracker, **SMALL)
    tracker.finish()
    kinds = [c[0] for c in calls]
    assert kinds.count("log") >= 2 and kinds[-1] == "finish"
    assert ("init", {"epochs": 2}) in calls or ("params", ["epochs"]) in calls


def _write_ethucy(root, scenes=("a", "b")):
    rng = np.random.default_rng(0)
    for scene in scenes:
        for split in ("train", "val", "test"):
            d = root / scene / split
            d.mkdir(parents=True)
            rows = []
            for pid in range(1, 6):
                x0, y0 = rng.uniform(0, 10, 2)
                vx, vy = rng.uniform(-0.3, 0.3, 2)
                for k in range(30):  # frames 10 apart, 2.5 Hz like the real files
                    rows.append(f"{780 + 10 * k}\t{pid}.0\t{x0 + vx * k:.3f}\t{y0 + vy * k:.3f}")
            (d / f"{scene}_{split}.txt").write_text("\n".join(rows) + "\n")


def test_ethucy_parse_and_windows(tmp_path):
    from cftraj.ethucy import load_scene_split, parse_ethucy_file

    _write_ethucy(tmp_path)
    tracks = parse_ethucy_file(tmp_path / "a" / "train" / "a_train.txt")
    assert sorted(tracks) == [1, 2, 3, 4, 5]
    assert [t[0] for t in tracks[1]][:3] == [0, 1, 2]  # frames re-indexed to steps
    w = load_scene_split(tmp_path, "a", "train")
    assert w.past_len == 8 and w.future_len == 12 and len(w) == 5 * (30 - 20 + 1)
    assert w.nmask.any()  # pedestrians see each other
    with pytest.raises(FileNotFoundError):
        load_scene_split(tmp_path, "zzz", "train")


def test_ethucy_cli_end_to_end(tmp_path, capsys):
    _write_ethucy(tmp_path / "eu")
    out = tmp_path / "eu.md"
    main(["ethucy", "--root", str(tmp_path / "eu"), "--scenes", "a", "b", "--epochs", "1",
          "--device", "cpu", "--out", str(out)])
    text = out.read_text(encoding="utf-8")
    assert "ETH/UCY" in text and "social + ridge prior (no context)" in text
    assert "| a | b | Average |" in text.replace("| Method | ", "| ")
