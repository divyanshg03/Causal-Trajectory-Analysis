import math
from pathlib import Path

import numpy as np
import pytest
import torch

from cftraj import baselines
from cftraj.__main__ import build_parser, seq_from_path
from cftraj.analyze import analyze_pair
from cftraj.counterfactual import apply_intervention, lateral_shift, scale_speed
from cftraj.data import (
    SPLITS,
    Windows,
    attach_neighbors,
    build_trajectories,
    clean_trajectories,
    concat_windows,
    create_windows,
    group_indices,
    load_windows,
    parse_kitti_labels,
)
from cftraj.evaluate import parse_ckpt_specs
from cftraj.metrics import ade_fde, error_by_horizon
from cftraj.model import (
    TrajectoryLSTM,
    TrajectorySocial,
    TrajectoryTransformer,
    load_model,
    predict_numpy,
    predict_windows,
    save_checkpoint,
)
from cftraj.pairs import all_pairs, find_interacting_pair
from cftraj.risk import (
    assess_pair,
    build_scene_graph,
    closing_speed,
    explain,
    min_distance,
    risk_from_ttc,
    time_to_collision,
)
from cftraj.train import train_model
from cftraj.validity import conflict_metrics, context_ablation, speed_response


def label_line(frame, tid, x, z, typ="Car"):
    # frame track type trunc occl alpha | bbox(4) | h w l | x y z | rot_y
    return f"{frame} {tid} {typ} 0 0 0 0 0 10 10 1.5 1.6 4.0 {x} 1.0 {z} 0.0\n"


def dontcare_line(frame):
    return f"{frame} -1 DontCare -1 -1 -10 1 1 2 2 -1000 -1000 -1000 -10 -1 -1 -1\n"


@pytest.fixture
def label_file(tmp_path):
    lines = []
    for f in range(30):  # two tracks approaching each other along x, plus DontCare
        lines.append(label_line(f, 0, -10 + 0.5 * f, 20.0))
        lines.append(label_line(f, 1, 10 - 0.5 * f, 20.0))
        lines.append(dontcare_line(f))
    p = tmp_path / "0000.txt"
    p.write_text("".join(lines))
    return p


def make_windows(starts, tids, last_points, seqs=None):
    n = len(starts)
    X = np.zeros((n, 5, 2), dtype=np.float32)
    X[:, -1, :] = last_points
    seqs = np.zeros(n, dtype=np.int64) if seqs is None else np.array(seqs)
    return Windows(X, np.zeros((n, 3, 2), np.float32), np.array(tids), np.array(starts), seqs)


# ---- data ---------------------------------------------------------------

def test_splits_are_disjoint_and_cover_21_sequences():
    all_ids = [s for ids in SPLITS.values() for s in ids]
    assert sorted(all_ids) == list(range(21))
    assert SPLITS["test"] == (18, 19, 20)


def test_parser_reads_3d_location_and_skips_dontcare(label_file):
    data = parse_kitti_labels(label_file)
    assert all(o["type"] != "DontCare" for objs in data.values() for o in objs)
    assert data[0][0]["location"] == [-10.0, 1.0, 20.0]
    assert build_trajectories(data)[0][0] == (0, -10.0, 20.0)  # (frame, x, z) in meters


def test_windows_cover_every_position(label_file):
    traj = clean_trajectories(build_trajectories(parse_kitti_labels(label_file)))
    w = create_windows(traj, past_len=5, future_len=3, seq=7)
    assert len(w) == 2 * (30 - 8 + 1)
    assert w.X.shape[1:] == (5, 2) and w.Y.shape[1:] == (3, 2)
    assert (w.seqs == 7).all()
    assert (w.past_len, w.future_len) == (5, 3)


def test_windows_skip_frame_gaps():
    traj = {0: [(f, float(f), 0.0) for f in [0, 1, 2, 3, 4, 10, 11, 12, 13, 14]]}
    assert len(create_windows(traj, past_len=5, future_len=3)) == 0


def test_group_indices():
    groups = group_indices(np.array([5, 3, 5, 3, 9]))
    assert sorted(sorted(g.tolist()) for g in groups) == [[0, 2], [1, 3], [4]]
    assert group_indices(np.array([], dtype=np.int64)) == []


def test_neighbors_are_nearest_cooccurring_and_masked(label_file):
    w = load_windows(label_file, 5, 3)
    assert w.nbr.shape == (len(w), 8, 5, 2)
    assert (w.nmask.sum(1) == 1).all()  # exactly one other agent per frame group
    for i in range(len(w)):
        j = np.flatnonzero((w.starts == w.starts[i]) & (w.tids != w.tids[i]))[0]
        assert w.ntids[i, 0] == w.tids[j]
        assert np.allclose(w.nbr[i, 0], w.X[j])
    assert (w.ntids[~w.nmask] == -1).all()


def test_neighbor_radius_and_k():
    X = np.zeros((3, 5, 2), np.float32)
    X[:, -1] = [[0, 0], [5, 0], [100, 0]]
    w = Windows(X, np.zeros((3, 3, 2), np.float32), np.arange(3), np.zeros(3, np.int64),
                np.zeros(3, np.int64))
    attach_neighbors(w, k=1, radius=10.0)
    assert w.nmask[0, 0] and w.ntids[0, 0] == 1  # nearest within radius
    assert not w.nmask[2].any()  # the far agent has none


def test_concat_windows(label_file):
    a = load_windows(label_file, 5, 3, seq=0)
    b = load_windows(label_file, 5, 3, seq=1)
    c = concat_windows([a, b])
    assert len(c) == len(a) + len(b) and c.nbr.shape[0] == len(c)
    assert set(np.unique(c.seqs)) == {0, 1}


def test_load_windows_missing_file():
    with pytest.raises(FileNotFoundError):
        load_windows("does/not/exist.txt")


# ---- counterfactual -----------------------------------------------------

def test_scale_speed_first_anchor():
    traj = np.array([[0, 0], [1, 0], [3, 0]], dtype=np.float32)
    assert np.allclose(scale_speed(traj, 1.0, anchor="first"), traj)
    assert np.allclose(scale_speed(traj, 2.0, anchor="first"), [[0, 0], [2, 0], [6, 0]])
    assert np.allclose(scale_speed(traj, 0.0, anchor="first"), [[0, 0]] * 3)


def test_scale_speed_last_anchor_keeps_current_position():
    traj = np.array([[0, 0], [1, 0], [3, 0]], dtype=np.float32)
    out = scale_speed(traj, 2.0)  # default anchor="last"
    assert np.allclose(out[-1], traj[-1])
    assert np.allclose(np.diff(out, axis=0), 2 * np.diff(traj, axis=0))
    with pytest.raises(ValueError):
        scale_speed(traj, 2.0, anchor="middle")


def test_scale_speed_batched_matches_single():
    batch = np.random.default_rng(0).normal(size=(4, 6, 2)).astype(np.float32)
    out = scale_speed(batch, 1.7)
    assert out.shape == batch.shape
    for k in range(4):
        assert np.allclose(out[k], scale_speed(batch[k], 1.7), atol=1e-5)


def test_lateral_shift_keeps_current_position_and_adds_sideways_motion():
    traj = np.array([[0, 0], [0, 1], [0, 2], [0, 3]], dtype=np.float32)
    out = lateral_shift(traj, 3.0)
    assert np.allclose(out[-1], traj[-1])
    assert np.allclose(out[:, 1], traj[:, 1])  # forward motion untouched
    assert np.allclose(np.diff(out[:, 0]), 1.0)  # drifted 3 m sideways over 3 steps


def test_apply_intervention_dispatch():
    traj = np.array([[0, 0], [1, 0], [3, 0]], dtype=np.float32)
    assert np.allclose(apply_intervention(traj, "speed", 2.0), scale_speed(traj, 2.0))
    assert np.allclose(apply_intervention(traj, "lateral", 1.0), lateral_shift(traj, 1.0))
    with pytest.raises(ValueError):
        apply_intervention(traj, "teleport", 1.0)


# ---- risk ---------------------------------------------------------------

def head_on(n=30, gap0=40.0, speed=1.0):
    """Two agents on the x axis closing at 2 * speed m per frame."""
    t = np.arange(n)[:, None]
    a = np.hstack([-gap0 / 2 + speed * t, np.zeros((n, 1))])
    b = np.hstack([gap0 / 2 - speed * t, np.zeros((n, 1))])
    return a, b


def test_ttc_first_collision_inside_horizon():
    a, b = head_on(n=30, gap0=20.0, speed=1.0)  # gap = 20 - 2t m, first < 2 m at index 10
    ttc = time_to_collision(a, b)
    assert ttc == pytest.approx(1.1)  # index 10 is 11 frames ahead
    assert assess_pair(a, b)["risk"] == "HIGH"


def test_ttc_extrapolates_when_still_approaching():
    a, b = head_on(n=10, gap0=60.0, speed=1.0)  # gap at end 60-18 = 42 m, closing 2 m/frame
    ttc = time_to_collision(a, b)
    assert ttc == pytest.approx(1.0 + (42.0 - 2.0) / 2.0 / 10.0)  # 3.0 s


def test_ttc_infinite_when_not_approaching():
    far = np.stack([np.linspace(0, 1, 10), np.zeros(10)], axis=1)
    far2 = far + [30.0, 0.0]
    assert time_to_collision(far, far2) == math.inf  # parallel, never meet
    assert assess_pair(far, far2)["risk"] == "SAFE"


def test_overlapping_agents_are_critical_not_low():
    p = np.zeros((10, 2))
    scene = assess_pair(p, p)  # distance 0: used to be rated LOW
    assert scene["risk"] == "CRITICAL"
    assert scene["ttc_s"] == pytest.approx(0.1)


def test_mid_horizon_crossing_is_not_hidden_by_end_state():
    t = np.arange(20)
    a = np.stack([np.zeros(20), t * 1.0], axis=1)
    b = np.stack([np.where(t < 10, 0.5 * (t - 10), 0.5 * (t - 10)), t * 1.0], axis=1)
    b[:, 0] = 0.5 * np.abs(t - 10)  # touches A's path mid-horizon, then moves away
    scene = assess_pair(a, b)
    assert scene["min_distance_m"] < 1.0
    assert scene["risk"] in ("CRITICAL", "HIGH")


def test_risk_from_ttc_tiers():
    assert risk_from_ttc(0.5, 10.0) == "CRITICAL"
    assert risk_from_ttc(2.0, 10.0) == "HIGH"
    assert risk_from_ttc(4.0, 10.0) == "MEDIUM"
    assert risk_from_ttc(math.inf, 2.0) == "LOW"  # came within 5 m
    assert risk_from_ttc(math.inf, 50.0) == "SAFE"


def test_closing_speed_sign_and_short_trajectories():
    a = np.array([[0.0, 0.0], [0.1, 0.0]])  # moving right
    b = np.array([[1.0, 0.0], [0.9, 0.0]])  # moving left, ahead of A
    assert closing_speed(a, b) < 0
    assert closing_speed(a, np.array([[1.0, 0.0], [1.3, 0.0]])) > 0


def test_scene_graph_units_and_explanation():
    a, b = head_on(n=10, gap0=60.0, speed=1.0)
    s = build_scene_graph(a, b)
    assert s["closing_speed_mps"] == pytest.approx(-20.0)  # 2 m/frame * 10 Hz
    assert s["relation"] == "approaching"
    assert "TTC" in explain(s)
    safe = build_scene_graph(a, a + [100.0, 0.0])
    assert "keeping pace" in explain(safe)


def test_min_distance_is_time_aligned():
    a = np.array([[0.0, 0.0], [1.0, 0.0]])
    b = np.array([[0.0, 3.0], [1.0, 1.0]])
    assert min_distance(a, b) == pytest.approx(1.0)


# ---- pair selection -----------------------------------------------------

def test_pairs_require_same_start_frame():
    w = make_windows([0, 5, 0], [0, 1, 2], [[0, 0], [0.001, 0], [5, 5]])
    assert find_interacting_pair(w) == (0, 2)


def test_pairs_require_same_sequence():
    w = make_windows([0, 0], [0, 1], [[0, 0], [0, 0]], seqs=[0, 1])
    assert find_interacting_pair(w) is None


def test_pairs_none_when_never_cooccurring():
    w = make_windows([0, 1, 2], [0, 1, 2], [[0, 0], [0, 0], [0, 0]])
    assert find_interacting_pair(w) is None


def test_pairs_max_distance():
    w = make_windows([0, 0], [0, 1], [[0, 0], [5, 5]])
    assert find_interacting_pair(w, max_distance=1.0) is None
    assert find_interacting_pair(w, max_distance=10.0) == (0, 1)


def test_all_pairs():
    w = make_windows([0, 0, 0, 1], [0, 1, 2, 3], [[0, 0], [5, 0], [50, 0], [1, 0]])
    assert sorted(map(tuple, all_pairs(w, 15.0).tolist())) == [(0, 1)]


# ---- baselines & metrics ------------------------------------------------

def linear_windows(n=20, past=10, future=10, seed=0):
    rng = np.random.default_rng(seed)
    p0 = rng.normal(size=(n, 1, 2))
    v = rng.normal(size=(n, 1, 2))
    t = np.arange(past + future)[None, :, None]
    full = (p0 + v * t).astype(np.float32)
    return full[:, :past], full[:, past:]


def test_baselines_exact_on_linear_motion():
    X, Y = linear_windows()
    assert np.allclose(baselines.constant_velocity(X, 10), Y, atol=1e-4)
    assert np.allclose(baselines.kalman(X, 10, q=1.0, r=0.01), Y, atol=1e-2)
    ridge = baselines.fit_ridge(X, Y, lam=1e-6)
    assert np.allclose(ridge(X), Y, atol=1e-2)
    assert baselines.stationary(X, 10).shape == Y.shape


def test_constant_velocity_short_past():
    X, Y = linear_windows(past=2, future=3)
    assert np.allclose(baselines.constant_velocity(X, 3), Y, atol=1e-4)  # k clipped to 1


def test_tune_kalman_returns_grid_value_with_unit_r():
    X, Y = linear_windows()
    q, r = baselines.tune_kalman(X, Y, qs=(0.1, 1.0, 10.0))
    assert q in (0.1, 1.0, 10.0) and r == 1.0


def test_metrics():
    true = np.zeros((2, 3, 2))
    pred = true.copy()
    pred[:, :, 0] = [[1, 2, 3], [1, 2, 3]]
    ade, fde = ade_fde(pred, true)
    assert ade == pytest.approx(2.0) and fde == pytest.approx(3.0)
    assert np.allclose(error_by_horizon(pred, true), [1, 2, 3])


# ---- model / training ---------------------------------------------------

@pytest.mark.parametrize("cls", [TrajectoryTransformer, TrajectoryLSTM, TrajectorySocial])
def test_model_shape_and_translation_invariance(cls):
    model = cls(past_len=6, future_len=4).eval()
    x = torch.randn(3, 6, 2)
    out = model(x)
    assert out.shape == (3, 4, 2)
    shift = torch.tensor([100.0, -50.0])
    assert torch.allclose(model(x + shift), out + shift, atol=1e-3)
    with pytest.raises(ValueError):
        model(torch.zeros(1, 7, 2))


def test_model_rejects_too_short_past():
    with pytest.raises(ValueError, match="at least 2"):
        TrajectoryTransformer(past_len=1)


def test_social_model_uses_neighbors_and_is_translation_invariant():
    torch.manual_seed(0)
    model = TrajectorySocial(past_len=5, future_len=3).eval()
    x = torch.randn(2, 5, 2)
    nbr = torch.randn(2, 4, 5, 2)
    mask = torch.tensor([[True, True, False, False], [True, False, False, False]])
    with_nbr = model(x, nbr, mask)
    no_nbr = model(x, nbr, torch.zeros_like(mask))
    assert not torch.allclose(with_nbr, no_nbr, atol=1e-6)
    shift = torch.tensor([30.0, 7.0])
    assert torch.allclose(model(x + shift, nbr + shift, mask), with_nbr + shift, atol=1e-3)
    assert model.last_attn.shape == (2, 4)
    assert torch.allclose(model.last_attn[~mask], torch.zeros(()), atol=1e-6)  # padding ignored
    assert torch.isfinite(model(x, nbr, torch.zeros_like(mask))).all()  # all masked: no NaN


def test_residual_model_starts_near_constant_velocity():
    X, _ = linear_windows(past=6, future=4)
    model = TrajectoryTransformer(past_len=6, future_len=4, residual=True).eval()
    for p in model.fc.parameters():
        torch.nn.init.zeros_(p)
    assert np.allclose(predict_numpy(model, X), baselines.constant_velocity(X, 4), atol=1e-4)


def test_checkpoint_roundtrip_and_legacy_error(tmp_path):
    model = TrajectoryTransformer(past_len=5, future_len=3).eval()
    path = tmp_path / "m.pt"
    save_checkpoint(path, model, "transformer", {"past_len": 5, "future_len": 3})
    loaded = load_model(path)
    x = np.random.randn(2, 5, 2).astype(np.float32)
    assert np.allclose(predict_numpy(model, x), predict_numpy(loaded, x), atol=1e-6)

    legacy = tmp_path / "old.pth"
    torch.save(model.state_dict(), legacy)
    with pytest.raises(ValueError, match="legacy"):
        load_model(legacy)


@pytest.mark.parametrize("arch,kw", [
    ("transformer", {"d_model": 16, "nhead": 2, "num_layers": 1, "dim_ff": 32}),
    ("social", {"d_model": 16, "nhead": 2, "num_layers": 1, "dim_ff": 32}),
])
def test_train_smoke_saves_checkpoint(tmp_path, label_file, arch, kw):
    w = load_windows(label_file, 5, 3)
    out = tmp_path / "t.pt"
    hist = train_model(arch, w, w, out, epochs=3, batch_size=16, device="cpu", log_every=99,
                       speed_aug=(0.8, 1.2), **kw)
    assert out.is_file() and (tmp_path / "t.pt.history.json").is_file()
    assert len(hist["val_ade"]) == 3 and np.isfinite(hist["best_val_ade"])
    assert load_model(out).future_len == 3


def test_train_rejects_zero_epochs(tmp_path, label_file):
    w = load_windows(label_file, 5, 3)
    with pytest.raises(ValueError, match="epochs"):
        train_model("transformer", w, w, tmp_path / "x.pt", epochs=0, device="cpu")


def test_train_fails_cleanly_when_validation_is_never_finite(tmp_path, label_file):
    w = load_windows(label_file, 5, 3)
    bad = Windows(w.X * np.nan, w.Y, w.tids, w.starts, w.seqs, w.nbr, w.nmask, w.ntids)
    with pytest.raises(RuntimeError, match="finite"):
        train_model("transformer", w, bad, tmp_path / "x.pt", epochs=2, device="cpu",
                    log_every=99)


def test_parse_ckpt_specs():
    assert parse_ckpt_specs(["a=x.pt,y.pt", "b=z.pt"]) == {"a": ["x.pt", "y.pt"], "b": ["z.pt"]}
    with pytest.raises(ValueError):
        parse_ckpt_specs(["nopath"])


def test_cli_helpers():
    assert seq_from_path("data/training/label_02/0007.txt") == 7
    assert seq_from_path("custom_labels.txt") == 0
    args = build_parser().parse_args(["analyze"])
    assert args.model == "checkpoints/social_h30.pt" and args.seq is None


# ---- validity -----------------------------------------------------------

def tiny_social(label_file):
    torch.manual_seed(0)
    windows = load_windows(label_file, 5, 3)
    model = TrajectorySocial(past_len=5, future_len=3, d_model=16, nhead=2, num_layers=1,
                             dim_ff=32).eval()
    return model, windows


def test_speed_response_reports_monotonicity(label_file):
    model, windows = tiny_social(label_file)
    res = speed_response(model, windows, n=20)
    assert 0.0 <= res["monotone"] <= 1.0 and res["n"] == 20
    # a constant-velocity-like model must respond ~linearly to speed
    cv_like = TrajectoryTransformer(past_len=5, future_len=3).eval()
    for p in cv_like.fc.parameters():
        torch.nn.init.zeros_(p)
    res = speed_response(cv_like, windows, n=20)
    assert res["monotone"] == 1.0
    assert res["travel_at_zero_m"] < 1e-3


def test_context_ablation_only_for_social(label_file):
    model, windows = tiny_social(label_file)
    out = context_ablation(model, windows)
    assert set(out) == {"ade", "fde", "ade_no_context", "fde_no_context"}
    assert context_ablation(TrajectoryLSTM(past_len=5, future_len=3), windows) is None


def test_conflict_metrics_perfect_and_stationary():
    # two agents 10 m apart that converge to the same point
    X = np.zeros((2, 5, 2), np.float32)
    X[0, -1], X[1, -1] = [-5, 0], [5, 0]
    Y = np.zeros((2, 8, 2), np.float32)
    Y[0] = np.linspace([-5, 0], [0, 0], 8)
    Y[1] = np.linspace([5, 0], [0.5, 0], 8)
    w = Windows(X, Y, np.array([0, 1]), np.zeros(2, np.int64), np.zeros(2, np.int64))
    perfect = conflict_metrics(w, Y)
    assert perfect["n_pairs"] == 1 and perfect["n_positive"] == 1
    assert perfect["distance"]["f1"] == 1.0
    stay = np.repeat(X[:, -1:], 8, axis=1)  # "stationary" predicts no conflict
    assert conflict_metrics(w, stay)["distance"]["recall"] == 0.0


def test_conflict_metrics_skips_already_close_pairs():
    X = np.zeros((2, 5, 2), np.float32)
    X[1, -1] = [1.0, 0.0]  # already within the 3 m radius now
    Y = np.zeros((2, 4, 2), np.float32)
    w = Windows(X, Y, np.array([0, 1]), np.zeros(2, np.int64), np.zeros(2, np.int64))
    assert conflict_metrics(w, Y)["n_pairs"] == 0


# ---- end to end ---------------------------------------------------------

def test_analyze_pair_end_to_end_with_social_model(label_file):
    model, windows = tiny_social(label_file)
    res = analyze_pair(model, windows, "speed", 2.0)
    assert res is not None
    assert res["track_ids"][0] != res["track_ids"][1]
    assert res["original"]["risk"] in {"CRITICAL", "HIGH", "MEDIUM", "LOW", "SAFE"}
    assert res["cf_pred_a"].shape == (3, 2)
    assert res["attention_a"].shape == (8,)
    # social model: B reacts to the intervened A
    assert not np.allclose(res["cf_pred_b"], res["pred_b"])
    # masking the scene context removes B's reaction
    masked = analyze_pair(model, windows, "speed", 2.0, mask_neighbors=True)
    assert np.allclose(masked["cf_pred_b"], masked["pred_b"], atol=1e-5)


def test_analyze_pair_plain_model_and_lateral(label_file):
    windows = load_windows(label_file, 5, 3)
    model = TrajectoryTransformer(past_len=5, future_len=3).eval()
    res = analyze_pair(model, windows, "lateral", 2.0)
    assert res["attention_a"] is None
    assert np.allclose(res["cf_pred_b"], res["pred_b"])  # plain models ignore other agents
    assert not np.allclose(res["cf_pred_a"], res["pred_a"])


def test_predict_windows_uses_neighbors_only_for_social(label_file):
    model, windows = tiny_social(label_file)
    full = predict_windows(model, windows)
    masked = predict_windows(model, windows, mask_neighbors=True)
    assert full.shape == masked.shape and not np.allclose(full, masked)


def test_attention_plot_and_demo(tmp_path, label_file):
    from cftraj.viz import make_demo, plot_attention

    model, windows = tiny_social(label_file)
    attn = plot_attention(model, windows, 0, tmp_path / "att.png")
    assert (tmp_path / "att.png").is_file() and attn.shape == (8,)
    with pytest.raises(ValueError, match="social"):
        plot_attention(TrajectoryLSTM(past_len=5, future_len=3), windows, 0)

    import matplotlib.image as mpimg

    img_dir = tmp_path / "img"
    img_dir.mkdir()
    for f in range(30):
        mpimg.imsave(img_dir / f"{f:06d}.png", np.zeros((20, 60, 3), np.float32))
    n = make_demo(model, windows, label_file, img_dir, tmp_path / "d.gif",
                  first_frame=4, n_frames=5, step=2)
    assert n == 5 and (tmp_path / "d.gif").is_file()


# ---- demo app -----------------------------------------------------------

@pytest.mark.skipif(
    not (Path("data/training/label_02/0018.txt").is_file()
         and Path("checkpoints/social_h30.pt").is_file()),
    reason="needs KITTI labels and the shipped checkpoint",
)
def test_streamlit_app_runs():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(Path(__file__).parent.parent / "app" / "streamlit_app.py"),
                          default_timeout=120).run()
    assert not at.exception
    at.radio[0].set_value("lateral").run()
    assert not at.exception and len(at.subheader) == 1


# ---- ridge prior --------------------------------------------------------

def test_ridge_initialised_model_reproduces_ridge():
    X, Y = linear_windows(n=50, past=6, future=4, seed=1)
    ridge = baselines.fit_ridge(X, Y, lam=1e-3)
    model = TrajectoryTransformer(past_len=6, future_len=4, residual=False,
                                  linear_skip=True).eval()
    model.init_skip(ridge.W)
    for p in model.fc.parameters():
        torch.nn.init.zeros_(p)
    assert np.allclose(predict_numpy(model, X), ridge(X), atol=1e-4)


def test_train_with_ridge_prior_never_worse_than_ridge_on_val(tmp_path, label_file):
    w = load_windows(label_file, 5, 3)
    ridge = baselines.fit_ridge(w.X, w.Y)
    ridge_ade = ade_fde(ridge(w.X), w.Y)[0]
    hist = train_model("social", w, w, tmp_path / "rp.pt", epochs=3, batch_size=16,
                       device="cpu", log_every=99, d_model=16, nhead=2, num_layers=1, dim_ff=32,
                       linear_skip=True, residual=False)
    assert hist["best_val_ade"] <= ridge_ade + 1e-3  # epoch 0 (pure ridge) is a candidate
    assert load_model(tmp_path / "rp.pt").skip is not None


def test_find_escalation_returns_a_risk_increase_or_none(label_file):
    from cftraj.analyze import find_escalation
    from cftraj.risk import RISK_ORDER

    windows = load_windows(label_file, 5, 3)
    model = TrajectoryTransformer(past_len=5, future_len=3).eval()
    res = find_escalation(model, windows, "speed", 3.0)
    if res is not None:
        o, c = res["original"]["risk"], res["counterfactual"]["risk"]
        assert RISK_ORDER.index(c) > RISK_ORDER.index(o)
    assert find_escalation(model, windows, "speed", 1.0) is None  # identity never escalates
