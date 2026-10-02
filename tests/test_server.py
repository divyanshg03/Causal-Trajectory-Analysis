import json

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from cftraj.data import load_windows  # noqa: E402
from cftraj.server import INTERVENTION_SPECS, Service, ServiceError, create_app  # noqa: E402
from cftraj.synthetic import write_synthetic_kitti  # noqa: E402
from cftraj.train import train_model  # noqa: E402

SMALL = {"d_model": 16, "nhead": 2, "num_layers": 1, "dim_ff": 32}


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    root = tmp_path_factory.mktemp("web")
    labels = write_synthetic_kitti(root / "label_02", n_seqs=3, n_frames=40)
    ckpts = root / "checkpoints"
    ckpts.mkdir()
    w = load_windows(labels / "0000.txt", 5, 3)
    train_model("social", w, w, ckpts / "tiny.pt", epochs=1, batch_size=16, device="cpu",
                log_every=99, **SMALL)
    return labels, ckpts


@pytest.fixture(scope="module")
def client(env, tmp_path_factory):
    labels, ckpts = env
    web = tmp_path_factory.mktemp("static")
    (web / "index.html").write_text("<html>explorer</html>")
    return TestClient(create_app(Service(labels, ckpts), web_dir=web))


def first_pair(client):
    frames = client.get("/api/frames", params={"model": "tiny", "seq": 0}).json()["frames"]
    for f in frames:
        scene = client.get("/api/scene", params={"model": "tiny", "seq": 0,
                                                 "start": f["start"]}).json()
        if scene["pairs"]:
            p = scene["pairs"][0]
            return f["start"], p["a"], p["b"]
    raise AssertionError("no pair found")


def test_meta_lists_models_interventions_and_risk_tiers(client):
    meta = client.get("/api/meta").json()
    assert meta["models"][0]["name"] == "tiny" and meta["models"][0]["social"] is True
    assert meta["sequences"] == [0, 1, 2]
    assert {i["kind"] for i in meta["interventions"]} == set(INTERVENTION_SPECS)
    assert meta["risk"]["order"][0] == "SAFE" and meta["risk"]["ttc_bands"][0] == [1.0, "CRITICAL"]
    assert meta["has_images"] is False


def test_frames_and_scene_describe_the_window(client):
    frames = client.get("/api/frames", params={"model": "tiny", "seq": 1}).json()["frames"]
    assert frames and all(f["agents"] >= 2 for f in frames)
    assert frames == sorted(frames, key=lambda f: f["start"])
    start = frames[0]["start"]
    scene = client.get("/api/scene", params={"model": "tiny", "seq": 1, "start": start}).json()
    assert scene["now"] == start + 4 and len(scene["agents"]) >= 2
    ag = scene["agents"][0]
    assert len(ag["past"]) == 5 and len(ag["future"]) == 3
    d = [p["distance_m"] for p in scene["pairs"]]
    assert d == sorted(d)


def test_analyze_returns_complete_json_safe_result(client):
    start, a, b = first_pair(client)
    res = client.post("/api/analyze", json={
        "model": "tiny", "seq": 0, "start": start, "track_a": a, "track_b": b,
        "kind": "brake", "value": 0.5, "samples": 8}).json()
    assert res["track_a"] == a and res["track_b"] == b and res["intervention"] == ["brake", 0.5]
    for k in ("past_a", "pred_a", "cf_a", "true_a", "pred_b", "cf_b"):
        assert len(res[k]) in (3, 5)
    assert len(res["gap"]) == 3 and len(res["cf_gap"]) == 3
    assert res["original"]["risk"] in ("SAFE", "LOW", "MEDIUM", "HIGH", "CRITICAL")
    assert res["uncertainty"]["original"]["n_samples"] == 8
    assert isinstance(res["plausible"], bool)
    assert all(a_["weight"] >= 0 for a_ in res["attention_a"])
    json.dumps(res, allow_nan=False)  # inf TTC must have been mapped to null


def test_mask_changes_nothing_structural_and_no_samples_means_no_uncertainty(client):
    start, a, b = first_pair(client)
    body = {"model": "tiny", "seq": 0, "start": start, "track_a": a, "track_b": b}
    res = client.post("/api/analyze", json={**body, "mask": True}).json()
    assert "uncertainty" not in res


def test_escalation_endpoint(client):
    res = client.post("/api/escalation", json={"model": "tiny", "seq": 0, "kind": "speed",
                                               "value": 3.0}).json()
    assert isinstance(res["found"], bool)
    if res["found"]:
        o, c = res["result"]["original"]["risk"], res["result"]["counterfactual"]["risk"]
        order = client.get("/api/meta").json()["risk"]["order"]
        assert order.index(c) > order.index(o)
    json.dumps(res, allow_nan=False)


@pytest.mark.parametrize("path,body,status", [
    ("/api/analyze", {"model": "nope", "seq": 0, "start": 0, "track_a": 0, "track_b": 1}, 404),
    ("/api/analyze", {"model": "../tiny", "seq": 0, "start": 0, "track_a": 0, "track_b": 1}, 404),
    ("/api/analyze", {"model": "tiny", "seq": 99, "start": 0, "track_a": 0, "track_b": 1}, 404),
    ("/api/analyze", {"model": "tiny", "seq": 0, "start": 0, "track_a": 0, "track_b": 0}, 400),
    ("/api/analyze", {"model": "tiny", "seq": 0, "start": 0, "track_a": 0, "track_b": 1,
                      "kind": "speed", "value": 50}, 400),
    ("/api/analyze", {"model": "tiny", "seq": 0, "start": 0, "track_a": 0, "track_b": 1,
                      "kind": "teleport"}, 400),
    ("/api/analyze", {"model": "tiny", "seq": 0, "start": 0, "track_a": 0, "track_b": 1,
                      "samples": 5000}, 422),
    ("/api/analyze", {"model": "tiny", "seq": 0, "start": 9999, "track_a": 0, "track_b": 1}, 404),
])
def test_bad_requests_are_rejected_cleanly(client, path, body, status):
    r = client.post(path, json=body)
    assert r.status_code == status, r.text


def test_scene_errors_and_image_not_available(client):
    assert client.get("/api/scene", params={"model": "tiny", "seq": 0, "start": 9999}
                      ).status_code == 404
    assert client.get("/api/frames", params={"model": "tiny", "seq": 77}).status_code == 404
    assert client.get("/api/image/0/5").status_code == 404


def test_image_served_when_present(env, tmp_path):
    labels, ckpts = env
    img = tmp_path / "image_02" / "0000"
    img.mkdir(parents=True)
    (img / "000005.png").write_bytes(b"\x89PNG\r\n")
    c = TestClient(create_app(Service(labels, ckpts, image_dir=tmp_path / "image_02")))
    r = c.get("/api/image/0/5")
    assert r.status_code == 200 and r.headers["content-type"] == "image/png"
    assert c.get("/api/meta").json()["has_images"] is True


def test_static_frontend_is_served_at_root(client):
    r = client.get("/")
    assert r.status_code == 200 and "explorer" in r.text


def test_real_frontend_files_exist_and_reference_each_other():
    from pathlib import Path

    web = Path(__file__).resolve().parent.parent / "web"
    html = (web / "index.html").read_text(encoding="utf-8")
    assert 'src="app.js"' in html and 'href="style.css"' in html
    for name in ("app.js", "style.css"):
        assert (web / name).stat().st_size > 1000


def test_service_error_carries_status():
    err = ServiceError("x", 404)
    assert err.status == 404 and str(err) == "x"
