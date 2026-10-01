import math

import numpy as np
import pytest
import torch

from cftraj.analyze import analyze_pair
from cftraj.counterfactual import scale_speed
from cftraj.data import (
    Windows,
    build_trajectories,
    clean_trajectories,
    create_windows,
    load_windows,
    normalize,
    parse_kitti_labels,
)
from cftraj.model import TrajectoryTransformer
from cftraj.pairs import find_interacting_pair
from cftraj.risk import (
    build_scene_graph,
    classify_risk,
    closing_speed,
    min_distance,
    ttc_seconds,
)


def label_line(frame, tid, x, y, typ="Car", half=10):
    # frame track type trunc occl alpha x1 y1 x2 y2 + 3D fields (ignored)
    return (
        f"{frame} {tid} {typ} 0 0 0 {x - half} {y - half} {x + half} {y + half} "
        "1 1 1 0 0 0 0\n"
    )


@pytest.fixture
def label_file(tmp_path):
    lines = []
    for f in range(12):  # two tracks moving side by side, one DontCare
        lines.append(label_line(f, 0, 100 + 5 * f, 200))
        lines.append(label_line(f, 1, 140 - 5 * f, 200))
        lines.append(label_line(f, -1, 10, 10, typ="DontCare"))
    p = tmp_path / "0000.txt"
    p.write_text("".join(lines))
    return p


# ---- data ---------------------------------------------------------------

def test_parser_skips_dontcare(label_file):
    data = parse_kitti_labels(label_file)
    assert all(o["type"] != "DontCare" for objs in data.values() for o in objs)
    assert len(data[0]) == 2


def test_windows_cover_every_position(label_file):
    traj = clean_trajectories(build_trajectories(parse_kitti_labels(label_file)))
    w = create_windows(traj, past_len=5, future_len=3)
    # 12 frames, window of 8 -> 5 windows per track
    assert len(w) == 10
    assert w.X.shape == (10, 5, 2) and w.Y.shape == (10, 3, 2)


def test_windows_skip_frame_gaps():
    traj = {0: [(f, float(f), 0.0) for f in [0, 1, 2, 3, 4, 10, 11, 12, 13, 14]]}
    assert len(create_windows(traj, past_len=5, future_len=3)) == 0


def test_normalize_does_not_mutate():
    arr = np.array([[[1242.0, 375.0]]], dtype=np.float32)
    out = normalize(arr)
    assert arr[0, 0, 0] == 1242.0
    assert np.allclose(out, 1.0)


def test_load_windows_missing_file():
    with pytest.raises(FileNotFoundError):
        load_windows("does/not/exist.txt")


# ---- counterfactual -----------------------------------------------------

def test_scale_speed_identity_and_double():
    traj = np.array([[0, 0], [1, 0], [3, 0]], dtype=np.float32)
    assert np.allclose(scale_speed(traj, 1.0), traj)
    assert np.allclose(scale_speed(traj, 2.0), [[0, 0], [2, 0], [6, 0]])
    assert np.allclose(scale_speed(traj, 0.0), [[0, 0]] * 3)


# ---- risk ---------------------------------------------------------------

def test_ttc_and_risk_tiers():
    assert ttc_seconds(1.0, 0.1) == math.inf
    # 0.1 units away closing at 0.1/frame = 1 frame = 0.1 s -> CRITICAL
    assert ttc_seconds(0.1, -0.1) == pytest.approx(0.1)
    assert classify_risk(0.1, -0.1) == "CRITICAL"
    assert classify_risk(0.1, -0.005) == "HIGH"  # 2 s
    assert classify_risk(0.1, -0.003) == "MEDIUM"  # ~3.3 s
    assert classify_risk(1.0, 0.0) == "SAFE"
    assert classify_risk(0.01, 0.0) == "LOW"


def test_closing_speed_sign():
    a = np.array([[0.0, 0.0], [0.1, 0.0]])  # moving right
    b = np.array([[1.0, 0.0], [0.9, 0.0]])  # moving left, ahead of A
    assert closing_speed(a, b) < 0
    assert build_scene_graph(a, b)["relation"] == "approaching"
    assert build_scene_graph(b, a)["relation"] == "approaching"
    b_away = np.array([[1.0, 0.0], [1.1, 0.0]])
    assert closing_speed(a, np.vstack([b_away[0], b_away[1]])) > 0


def test_min_distance_is_time_aligned():
    a = np.array([[0.0, 0.0], [1.0, 0.0]])
    b = np.array([[0.0, 3.0], [1.0, 1.0]])
    assert min_distance(a, b) == pytest.approx(1.0)


# ---- pair selection -----------------------------------------------------

def make_windows(starts, tids, last_points):
    n = len(starts)
    X = np.zeros((n, 5, 2), dtype=np.float32)
    X[:, -1, :] = last_points
    return Windows(X, np.zeros((n, 3, 2), np.float32), np.array(tids), np.array(starts))


def test_pairs_require_same_start_frame():
    # nearest points belong to windows at different frames -> must not pair
    w = make_windows([0, 5, 0], [0, 1, 2], [[0, 0], [0.001, 0], [0.5, 0.5]])
    assert find_interacting_pair(w) == (0, 2)


def test_pairs_none_when_never_cooccurring():
    w = make_windows([0, 1, 2], [0, 1, 2], [[0, 0], [0, 0], [0, 0]])
    assert find_interacting_pair(w) is None


def test_pairs_max_distance():
    w = make_windows([0, 0], [0, 1], [[0, 0], [0.5, 0.5]])
    assert find_interacting_pair(w, max_distance=0.1) is None
    assert find_interacting_pair(w, max_distance=1.0) == (0, 1)


# ---- model / end to end -------------------------------------------------

def test_model_output_shape():
    model = TrajectoryTransformer()
    assert model(torch.zeros(4, 5, 2)).shape == (4, 3, 2)
    with pytest.raises(ValueError):
        model(torch.zeros(1, 11, 2))


def test_analyze_pair_end_to_end(label_file):
    torch.manual_seed(0)
    windows = load_windows(label_file)
    res = analyze_pair(TrajectoryTransformer().eval(), windows, factor=2.0)
    assert res is not None
    assert res["track_ids"][0] != res["track_ids"][1]
    assert res["original"]["risk"] in {"CRITICAL", "HIGH", "MEDIUM", "LOW", "SAFE"}
    assert res["cf_pred_a"].shape == (3, 2)
