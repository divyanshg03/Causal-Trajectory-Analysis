"""Web backend for the counterfactual explorer (``python -m cftraj serve``).

:class:`Service` holds the data/model caches and returns plain JSON-able dicts, so it is
testable without HTTP; :func:`create_app` wraps it in a small FastAPI app that also serves
the static frontend from ``web/``. Models and label files are only ever looked up by name
from the configured directories (no client-supplied paths).
"""

import math
from pathlib import Path

import numpy as np

from .analyze import analyze_pair, find_escalation
from .counterfactual import INTERVENTIONS, MAX_ACCEL
from .data import KITTI_FPS, group_indices, load_windows
from .model import load_model
from .pairs import all_pairs
from .risk import COLLISION_RADIUS, NEAR_DISTANCE, RISK_ORDER, TTC_BANDS, gap_series
from .train import pick_device

PAIR_RADIUS = 40.0  # pairs offered in the UI (m between last observed points)

# slider range, step, default, unit, label for each intervention
INTERVENTION_SPECS = {
    "speed": {"label": "Scale speed", "min": 0.0, "max": 3.0, "step": 0.1, "default": 2.0,
              "unit": "x", "help": "A had been moving this many times faster (0 = standing still)."},
    "lateral": {"label": "Drift sideways", "min": -6.0, "max": 6.0, "step": 0.5, "default": 2.0,
                "unit": "m", "help": "A looks like it drifted this far sideways to reach its position."},
    "brake": {"label": "Braking", "min": 0.0, "max": 1.0, "step": 0.05, "default": 0.8,
              "unit": "", "help": "A had been slowing down; 1 = coming to a stop right now."},
    "delay": {"label": "Late start", "min": 0, "max": 8, "step": 1, "default": 3,
              "unit": "frames", "help": "A sat still until this many frames before now."},
}


class ServiceError(ValueError):
    """A client error (unknown model, bad frame, ...); mapped to HTTP 400/404."""

    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


def _num(x):
    """JSON-safe float: inf/nan -> None."""
    x = float(x)
    return x if math.isfinite(x) else None


def _arr(a):
    return np.asarray(a, dtype=float).round(3).tolist()


def _scene_json(scene):
    out = {k: v for k, v in scene.items() if k not in ("object_A", "object_B")}
    for k in ("ttc_s", "distance_m", "min_distance_m", "closing_speed_mps"):
        out[k] = _num(out[k])
    return out


class Service:
    def __init__(self, label_dir="data/training/label_02", ckpt_dir="checkpoints",
                 image_dir=None, device="cpu"):
        self.label_dir = Path(label_dir)
        self.ckpt_dir = Path(ckpt_dir)
        self.image_dir = Path(image_dir) if image_dir else self.label_dir.parent / "image_02"
        self.device = pick_device(device)
        self._models, self._windows = {}, {}

    # -- lookups -----------------------------------------------------------
    def model_names(self):
        return sorted(p.stem for p in self.ckpt_dir.glob("*.pt"))

    def sequences(self):
        return sorted(int(p.stem) for p in self.label_dir.glob("*.txt") if p.stem.isdigit())

    def model(self, name):
        if name not in self.model_names():
            raise ServiceError(f"Unknown model '{name}'", 404)
        if name not in self._models:
            self._models[name] = load_model(self.ckpt_dir / f"{name}.pt", self.device)
        return self._models[name]

    def windows(self, seq, model):
        if seq not in self.sequences():
            raise ServiceError(f"Sequence {seq} not found in {self.label_dir}", 404)
        key = (seq, model.past_len, model.future_len)
        if key not in self._windows:
            self._windows[key] = load_windows(self.label_dir / f"{seq:04d}.txt", model.past_len,
                                              model.future_len, seq=seq)
        return self._windows[key]

    # -- API ----------------------------------------------------------------
    def meta(self):
        models = []
        for name in self.model_names():
            m = self.model(name)
            models.append({"name": name, "arch": m.arch, "past_len": m.past_len,
                           "future_len": m.future_len, "social": m.uses_neighbors,
                           "n_modes": m.n_modes, "horizon_s": m.future_len / KITTI_FPS})
        return {
            "models": models, "sequences": self.sequences(), "fps": KITTI_FPS,
            "interventions": [{"kind": k, **INTERVENTION_SPECS[k]} for k in INTERVENTIONS],
            "risk": {"order": list(RISK_ORDER), "ttc_bands": [list(b) for b in TTC_BANDS],
                     "collision_radius_m": COLLISION_RADIUS, "near_distance_m": NEAR_DISTANCE,
                     "max_accel_mps2": MAX_ACCEL},
            "has_images": self.image_dir.is_dir(),
        }

    def frames(self, model_name, seq):
        """Window start frames that have at least one pair, with agent counts."""
        m = self.model(model_name)
        w = self.windows(seq, m)
        out = []
        for g in group_indices(w.group_keys()):
            if len(g) < 2:
                continue
            out.append({"start": int(w.starts[g[0]]), "now": int(w.starts[g[0]]) + m.past_len - 1,
                        "agents": len(g)})
        out.sort(key=lambda r: r["start"])
        return {"seq": seq, "frames": out}

    def _group(self, w, start):
        idx = np.flatnonzero(w.starts == start)
        if len(idx) < 2:
            raise ServiceError(f"No two co-occurring agents at window start {start}", 404)
        return idx

    def scene(self, model_name, seq, start):
        """Agents of one window: observed past, true future, and candidate pairs."""
        m = self.model(model_name)
        w = self.windows(seq, m)
        idx = self._group(w, start)
        agents = [{"track": int(w.tids[i]), "past": _arr(w.X[i]), "future": _arr(w.Y[i])}
                  for i in idx]
        pairs = []
        for a, b in all_pairs(type("G", (), {
                "X": w.X[idx], "group_keys": lambda self: w.group_keys()[idx]})(), PAIR_RADIUS):
            i, j = idx[a], idx[b]
            d = float(np.linalg.norm(w.X[i, -1] - w.X[j, -1]))
            pairs.append({"a": int(w.tids[i]), "b": int(w.tids[j]), "distance_m": round(d, 2)})
        pairs.sort(key=lambda p: p["distance_m"])
        return {"seq": seq, "start": start, "now": start + m.past_len - 1, "agents": agents,
                "pairs": pairs}

    def _index(self, w, start, track):
        hit = np.flatnonzero((w.starts == start) & (w.tids == track))
        if len(hit) == 0:
            raise ServiceError(f"Track {track} not present at window start {start}", 404)
        return int(hit[0])

    def _check_intervention(self, kind, value):
        if kind not in INTERVENTION_SPECS:
            raise ServiceError(f"Unknown intervention '{kind}'")
        spec = INTERVENTION_SPECS[kind]
        if not (spec["min"] <= value <= spec["max"]):
            raise ServiceError(f"{kind} value must be within [{spec['min']}, {spec['max']}]")

    def _result_json(self, res, w):
        i, j = res["indices"]
        o, c = res["original"], res["counterfactual"]
        att = None
        if res["attention_a"] is not None:
            att = [{"track": int(t), "weight": round(float(x), 4)}
                   for t, x, ok in zip(w.ntids[i], res["attention_a"], w.nmask[i]) if ok]
        out = {
            "seq": res["seq"], "start": res["frame"], "track_a": res["track_ids"][0],
            "track_b": res["track_ids"][1], "intervention": list(res["intervention"]),
            "past_a": _arr(res["past_a"]), "past_b": _arr(res["past_b"]),
            "true_a": _arr(w.Y[i]), "true_b": _arr(w.Y[j]),
            "pred_a": _arr(res["pred_a"]), "pred_b": _arr(res["pred_b"]),
            "cf_a": _arr(res["cf_pred_a"]), "cf_b": _arr(res["cf_pred_b"]),
            "gap": _arr(gap_series(res["pred_a"], res["pred_b"])),
            "cf_gap": _arr(gap_series(res["cf_pred_a"], res["cf_pred_b"])),
            "original": {"risk": o["risk"], "scene": _scene_json(o["scene"]),
                         "explanation": o["explanation"]},
            "counterfactual": {"risk": c["risk"], "scene": _scene_json(c["scene"]),
                               "explanation": c["explanation"]},
            "plausible": res["plausible"], "max_accel_cf_mps2": _num(res["max_accel_cf_mps2"]),
            "attention_a": att,
        }
        if "uncertainty" in res:
            out["uncertainty"] = {
                k: {**u, "ttc_median_s": _num(u["ttc_median_s"]), "ttc_p10_s": _num(u["ttc_p10_s"])}
                for k, u in res["uncertainty"].items()}
        return out

    def analyze(self, model_name, seq, start, track_a, track_b, kind, value, mask=False,
                samples=0):
        if track_a == track_b:
            raise ServiceError("Pick two different agents")
        self._check_intervention(kind, value)
        if not 0 <= samples <= 200:
            raise ServiceError("samples must be between 0 and 200")
        m = self.model(model_name)
        w = self.windows(seq, m)
        pair = (self._index(w, start, track_a), self._index(w, start, track_b))
        res = analyze_pair(m, w, kind, value, pair=pair, mask_neighbors=mask, n_samples=samples)
        return self._result_json(res, w)

    def escalation(self, model_name, seq, kind, value, mask=False, samples=0):
        """The pair in this sequence whose risk the intervention raises the most."""
        self._check_intervention(kind, value)
        m = self.model(model_name)
        w = self.windows(seq, m)
        res = find_escalation(m, w, kind, value, mask_neighbors=mask, n_samples=samples)
        return {"found": res is not None,
                "result": None if res is None else self._result_json(res, w)}

    def image_path(self, seq, frame):
        if seq not in self.sequences():
            raise ServiceError("Unknown sequence", 404)
        p = self.image_dir / f"{seq:04d}" / f"{frame:06d}.png"
        if not p.is_file():
            raise ServiceError("Frame image not available", 404)
        return p


def create_app(service=None, web_dir=None, **service_kw):
    """FastAPI app: JSON API under ``/api`` plus the static frontend at ``/``."""
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel, Field

    svc = service or Service(**service_kw)
    app = FastAPI(title="cftraj counterfactual explorer", docs_url="/api/docs", redoc_url=None)

    class AnalyzeRequest(BaseModel):
        model: str
        seq: int
        start: int
        track_a: int
        track_b: int
        kind: str = "speed"
        value: float = 2.0
        mask: bool = False
        samples: int = Field(0, ge=0, le=200)

    class EscalationRequest(BaseModel):
        model: str
        seq: int
        kind: str = "speed"
        value: float = 2.0
        mask: bool = False
        samples: int = Field(0, ge=0, le=200)

    @app.exception_handler(ServiceError)
    async def _service_error(_, err: ServiceError):
        return JSONResponse({"detail": str(err)}, status_code=err.status)

    @app.exception_handler(FileNotFoundError)
    async def _missing(_, err: FileNotFoundError):
        return JSONResponse({"detail": str(err)}, status_code=404)

    @app.get("/api/meta")
    def meta():
        return svc.meta()

    @app.get("/api/frames")
    def frames(model: str, seq: int):
        return svc.frames(model, seq)

    @app.get("/api/scene")
    def scene(model: str, seq: int, start: int):
        return svc.scene(model, seq, start)

    @app.post("/api/analyze")
    def analyze(req: AnalyzeRequest):
        return svc.analyze(req.model, req.seq, req.start, req.track_a, req.track_b, req.kind,
                           req.value, req.mask, req.samples)

    @app.post("/api/escalation")
    def escalation(req: EscalationRequest):
        return svc.escalation(req.model, req.seq, req.kind, req.value, req.mask, req.samples)

    @app.get("/api/image/{seq}/{frame}")
    def image(seq: int, frame: int):
        try:
            return FileResponse(svc.image_path(seq, frame), media_type="image/png")
        except ServiceError as err:
            raise HTTPException(err.status, str(err)) from err

    web = Path(web_dir) if web_dir else Path(__file__).resolve().parent.parent / "web"
    if web.is_dir():
        app.mount("/", StaticFiles(directory=web, html=True), name="web")
    return app


def serve(host="127.0.0.1", port=8000, **service_kw):
    import uvicorn

    uvicorn.run(create_app(**service_kw), host=host, port=port)
