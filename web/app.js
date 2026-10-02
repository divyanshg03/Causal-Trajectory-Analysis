"use strict";
(() => {
  const $ = (id) => document.getElementById(id);
  const NS = "http://www.w3.org/2000/svg";
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const fmt = (x, d = 1) => (x === null || x === undefined ? "∞" : Number(x).toFixed(d));

  const S = {
    meta: null, model: null, seq: null, frames: [], start: null, scene: null,
    pair: null, kind: "speed", value: 2, mask: false, samples: false,
    result: null, t: 0, playing: false, view: null, token: 0, timer: null, raf: null,
  };

  // ---------- api ----------
  async function api(path, body) {
    const res = await fetch(path, body ? {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    } : undefined);
    if (!res.ok) {
      let msg = res.statusText;
      try { msg = (await res.json()).detail || msg; } catch (_) { /* keep status text */ }
      throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
    }
    return res.json();
  }
  function toast(msg) {
    const el = $("toast");
    el.textContent = msg; el.hidden = false;
    clearTimeout(toast.h); toast.h = setTimeout(() => { el.hidden = true; }, 4200);
  }
  const busy = (on) => { $("spinner").hidden = !on; $("escalate").disabled = on && S.escalating; };

  // ---------- state <-> url ----------
  function saveHash() {
    const p = new URLSearchParams({ m: S.model, s: S.seq, f: S.start ?? "", k: S.kind, v: S.value });
    if (S.pair) { p.set("a", S.pair.a); p.set("b", S.pair.b); }
    if (S.mask) p.set("mask", "1");
    if (S.samples) p.set("unc", "1");
    history.replaceState(null, "", "#" + p.toString());
  }
  function readHash() {
    const p = new URLSearchParams(location.hash.slice(1));
    return {
      m: p.get("m"), s: p.get("s") !== null ? Number(p.get("s")) : null,
      f: p.get("f") ? Number(p.get("f")) : null, k: p.get("k"), v: p.get("v") !== null ? Number(p.get("v")) : null,
      a: p.get("a") ? Number(p.get("a")) : null, b: p.get("b") ? Number(p.get("b")) : null,
      mask: p.get("mask") === "1", unc: p.get("unc") === "1",
    };
  }

  // ---------- init ----------
  async function init() {
    try { const th = localStorage.getItem("theme"); if (th) document.documentElement.dataset.theme = th; } catch (_) { /* ignore */ }
    let meta;
    try { meta = await api("/api/meta"); } catch (e) { return fatal(e.message); }
    S.meta = meta;
    if (!meta.models.length) return fatal("No checkpoints found. Run: python -m cftraj download");
    if (!meta.sequences.length) return fatal("No KITTI label files found. See 'Dataset setup' in the README.");
    const h = readHash();
    fill($("model"), meta.models.map((m) => [m.name, `${m.name} · ${m.horizon_s.toFixed(0)} s${m.social ? " · social" : ""}${m.n_modes > 1 ? " · " + m.n_modes + " modes" : ""}`]));
    fill($("seq"), meta.sequences.map((s) => [s, String(s).padStart(4, "0")]));
    S.model = meta.models.some((m) => m.name === h.m) ? h.m : meta.models[0].name;
    S.seq = meta.sequences.includes(h.s) ? h.s : (meta.sequences.includes(19) ? 19 : meta.sequences[0]);
    $("model").value = S.model; $("seq").value = S.seq;
    buildKinds();
    if (h.k && meta.interventions.some((i) => i.kind === h.k)) S.kind = h.k;
    S.value = h.v ?? kindSpec().default;
    S.mask = h.mask; S.samples = h.unc;
    $("mask").checked = S.mask; $("samples").checked = S.samples;
    renderLegend(); syncKind();
    bind();
    let [start, a, b] = [h.f, h.a, h.b];
    if (start === null) {  // no deep link: open on the pair this intervention escalates most
      try {
        $("spinner").hidden = false;
        const r = await api("/api/escalation", { model: S.model, seq: S.seq, kind: S.kind, value: S.value, mask: S.mask, samples: 0 });
        if (r.found) [start, a, b] = [r.result.start, r.result.track_a, r.result.track_b];
      } catch (_) { /* fall back to the middle of the sequence */ }
    }
    await loadFrames(start, a, b);
  }
  function fatal(msg) {
    const b = $("banner"); b.textContent = msg; b.hidden = false;
    showEmpty(msg);
  }
  function fill(sel, items) {
    sel.innerHTML = "";
    for (const [v, label] of items) { const o = document.createElement("option"); o.value = v; o.textContent = label; sel.appendChild(o); }
  }
  const model = () => S.meta.models.find((m) => m.name === S.model);
  const kindSpec = () => S.meta.interventions.find((i) => i.kind === S.kind);

  // ---------- data loading ----------
  async function loadFrames(wantStart, wantA, wantB) {
    busy(true);
    try {
      const r = await api(`/api/frames?model=${encodeURIComponent(S.model)}&seq=${S.seq}`);
      S.frames = r.frames;
      if (!S.frames.length) { S.scene = null; S.result = null; showEmpty("No frame in this sequence has two co-occurring agents with a full window."); renderAll(); return; }
      let idx = S.frames.findIndex((f) => f.start === wantStart);
      if (idx < 0) idx = Math.floor(S.frames.length / 2);
      $("frame").max = S.frames.length - 1; $("frame").value = idx;
      drawTicks();
      await loadScene(S.frames[idx].start, wantA, wantB);
    } catch (e) { toast(e.message); } finally { busy(false); }
  }
  async function loadScene(start, wantA, wantB) {
    S.start = start;
    const f = S.frames.find((x) => x.start === start);
    $("frame-label").textContent = `frame ${f.now}`;
    $("agents-label").textContent = `${f.agents} agents`;
    S.scene = await api(`/api/scene?model=${encodeURIComponent(S.model)}&seq=${S.seq}&start=${start}`);
    const pairs = S.scene.pairs;
    const keep = pairs.find((p) => p.a === wantA && p.b === wantB) || (S.pair && pairs.find((p) => p.a === S.pair.a && p.b === S.pair.b));
    S.pair = keep || pairs[0] || null;
    renderPairs();
    if (!S.pair) { S.result = null; showEmpty("No pair of agents within 40 m in this frame; try another frame."); renderAll(); return; }
    await runAnalyze();
  }
  let debounce = null;
  function scheduleAnalyze() { clearTimeout(debounce); debounce = setTimeout(runAnalyze, 140); }
  async function runAnalyze() {
    if (!S.pair) return;
    const token = ++S.token;
    busy(true);
    try {
      const r = await api("/api/analyze", {
        model: S.model, seq: S.seq, start: S.start, track_a: S.pair.a, track_b: S.pair.b,
        kind: S.kind, value: S.value, mask: S.mask, samples: S.samples ? 30 : 0,
      });
      if (token !== S.token) return;
      setResult(r);
    } catch (e) { if (token === S.token) toast(e.message); } finally { if (token === S.token) busy(false); }
  }
  function setResult(r) {
    const first = !S.result || S.result.track_a !== r.track_a || S.result.track_b !== r.track_b || S.result.start !== r.start;
    S.result = r;
    S.t = model().future_len;
    $("t").max = model().future_len; $("t").value = S.t;
    if (first) S.view = null;
    hideEmpty(); saveHash(); renderAll();
  }

  // ---------- rendering: lists / controls ----------
  function buildKinds() {
    const box = $("kinds"); box.innerHTML = "";
    for (const k of S.meta.interventions) {
      const b = document.createElement("button");
      b.type = "button"; b.textContent = k.label; b.dataset.kind = k.kind;
      b.setAttribute("role", "radio");
      b.addEventListener("click", () => { S.kind = k.kind; S.value = k.default; syncKind(); saveHash(); scheduleAnalyze(); });
      box.appendChild(b);
    }
  }
  function syncKind() {
    const spec = kindSpec();
    for (const b of $("kinds").children) b.setAttribute("aria-checked", String(b.dataset.kind === S.kind));
    const v = $("value");
    v.min = spec.min; v.max = spec.max; v.step = spec.step; v.value = S.value;
    $("value-label").textContent = spec.label;
    $("kind-help").textContent = spec.help;
    syncValueOut();
  }
  function syncValueOut() {
    const spec = kindSpec();
    const d = spec.step < 1 ? (spec.step < 0.1 ? 2 : 1) : 0;
    $("value-out").textContent = `${Number(S.value).toFixed(d)}${spec.unit ? " " + spec.unit : ""}`;
  }
  function renderPairs() {
    const ul = $("pairs"); ul.innerHTML = "";
    if (!S.scene || !S.scene.pairs.length) { const li = document.createElement("li"); li.className = "empty"; li.textContent = "No pair within 40 m"; ul.appendChild(li); return; }
    for (const p of S.scene.pairs) {
      const li = document.createElement("li");
      li.tabIndex = 0; li.setAttribute("role", "option");
      li.setAttribute("aria-selected", String(S.pair && p.a === S.pair.a && p.b === S.pair.b));
      li.innerHTML = `<span class="ab"><b>#${p.a}</b> &amp; <b>#${p.b}</b></span><span class="muted mono">${p.distance_m.toFixed(1)} m</span>`;
      const pick = () => { S.pair = p; renderPairs(); runAnalyze(); };
      li.addEventListener("click", pick);
      li.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pick(); } });
      ul.appendChild(li);
    }
    const sel = ul.querySelector('[aria-selected="true"]');
    if (sel) ul.scrollTop = Math.max(0, sel.offsetTop - ul.offsetTop - 8);
  }
  function drawTicks() {
    const c = $("frame-ticks"), dpr = window.devicePixelRatio || 1;
    const w = c.clientWidth || 260, h = 22;
    c.width = w * dpr; c.height = h * dpr;
    const g = c.getContext("2d"); g.scale(dpr, dpr); g.clearRect(0, 0, w, h);
    const n = S.frames.length; if (!n) return;
    const max = Math.max(...S.frames.map((f) => f.agents));
    g.fillStyle = css("--muted");
    S.frames.forEach((f, i) => {
      const x = 8 + (i / Math.max(1, n - 1)) * (w - 16), bh = 3 + 15 * (f.agents / max);
      g.globalAlpha = 0.55; g.fillRect(x - 0.5, h - bh, 1.2, bh);
    });
  }
  function renderLegend() {
    $("legend").innerHTML = [
      `<span><i style="border-color:var(--a)"></i>A (observed → predicted)</span>`,
      `<span><i style="border-color:var(--b)"></i>B</span>`,
      `<span><i class="dash" style="border-color:var(--cf)"></i>A counterfactual</span>`,
      `<span><i class="dash" style="border-color:var(--react)"></i>B reacts</span>`,
      `<span><i class="dash" style="border-color:var(--truth)"></i>ground truth</span>`,
    ].join("");
  }

  // ---------- rendering: BEV ----------
  const svg = $("bev");
  const el = (name, attrs = {}, parent) => {
    const n = document.createElementNS(NS, name);
    for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
    if (parent) parent.appendChild(n);
    return n;
  };
  function resultBounds() {
    const r = S.result, pts = [];
    for (const k of ["past_a", "past_b", "pred_a", "pred_b", "cf_a", "cf_b", "true_a", "true_b"]) pts.push(...r[k]);
    let x0 = Infinity, x1 = -Infinity, z0 = Infinity, z1 = -Infinity;
    for (const [x, z] of pts) { x0 = Math.min(x0, x); x1 = Math.max(x1, x); z0 = Math.min(z0, z); z1 = Math.max(z1, z); }
    return { x0, x1, z0, z1 };
  }
  function fitView() {
    const W = svg.clientWidth, H = svg.clientHeight, b = resultBounds();
    const pad = 40, spanX = Math.max(b.x1 - b.x0, 6), spanZ = Math.max(b.z1 - b.z0, 6);
    const s = Math.min((W - 2 * pad) / spanX, (H - 2 * pad) / spanZ);
    S.view = { s, ox: W / 2 - ((b.x0 + b.x1) / 2) * s, oy: H / 2 + ((b.z0 + b.z1) / 2) * s };
  }
  const px = (p) => [S.view.ox + p[0] * S.view.s, S.view.oy - p[1] * S.view.s];
  const poly = (pts) => pts.map((p) => px(p).map((v) => v.toFixed(1)).join(",")).join(" ");
  function niceStep(span) {
    const raw = span / 7, mag = Math.pow(10, Math.floor(Math.log10(raw))), f = raw / mag;
    return (f < 1.5 ? 1 : f < 3.5 ? 2 : f < 7.5 ? 5 : 10) * mag;
  }
  function drawGrid() {
    const W = svg.clientWidth, H = svg.clientHeight, { s, ox, oy } = S.view;
    const xa = (0 - ox) / s, xb = (W - ox) / s, zb = (oy - 0) / s, za = (oy - H) / s;
    const step = niceStep(Math.max(xb - xa, zb - za));
    const g = el("g", { "aria-hidden": "true" }, svg);
    for (let x = Math.ceil(xa / step) * step; x <= xb; x += step) {
      const X = ox + x * s;
      el("line", { x1: X, y1: 0, x2: X, y2: H, stroke: css("--grid"), "stroke-width": 1 }, g);
      const t = el("text", { x: X + 3, y: H - 5, fill: css("--muted"), "font-size": 10 }, g); t.textContent = `${+x.toFixed(2)} m`;
    }
    for (let z = Math.ceil(za / step) * step; z <= zb; z += step) {
      const Y = oy - z * s;
      el("line", { x1: 0, y1: Y, x2: W, y2: Y, stroke: css("--grid"), "stroke-width": 1 }, g);
      const t = el("text", { x: 4, y: Y - 3, fill: css("--muted"), "font-size": 10 }, g); t.textContent = `${+z.toFixed(2)} m`;
    }
  }
  function renderBev() {
    svg.innerHTML = "";
    if (!S.result) return;
    if (!S.view) fitView();
    const r = S.result, F = model().future_len, t = Math.min(S.t, F);
    drawGrid();
    const A = css("--a"), B = css("--b"), CF = css("--cf"), RE = css("--react"), TR = css("--truth"), OT = css("--other");

    if ($("show-others").checked && S.scene) {
      for (const ag of S.scene.agents) {
        if (ag.track === r.track_a || ag.track === r.track_b) continue;
        el("polyline", { points: poly(ag.past), fill: "none", stroke: OT, "stroke-width": 2, "stroke-linecap": "round" }, svg);
        el("polyline", { points: poly([ag.past[ag.past.length - 1], ...ag.future]), fill: "none", stroke: OT, "stroke-width": 1.5, "stroke-dasharray": "3 4" }, svg);
        const [x, y] = px(ag.past[ag.past.length - 1]);
        el("circle", { cx: x, cy: y, r: 3.5, fill: OT }, svg);
      }
    }
    const nowA = r.past_a[r.past_a.length - 1], nowB = r.past_b[r.past_b.length - 1];
    if ($("show-truth").checked) {
      el("polyline", { points: poly([nowA, ...r.true_a]), fill: "none", stroke: TR, "stroke-width": 1.8, "stroke-dasharray": "2 4", opacity: .9 }, svg);
      el("polyline", { points: poly([nowB, ...r.true_b]), fill: "none", stroke: TR, "stroke-width": 1.8, "stroke-dasharray": "2 4", opacity: .9 }, svg);
    }
    const path = (past, fut) => poly([past[past.length - 1], ...fut.slice(0, t)]);
    const reacts = r.cf_b.some((p, i) => Math.abs(p[0] - r.pred_b[i][0]) + Math.abs(p[1] - r.pred_b[i][1]) > 0.02);
    for (const [past, color] of [[r.past_a, A], [r.past_b, B]]) el("polyline", { points: poly(past), fill: "none", stroke: color, "stroke-width": 3, "stroke-linecap": "round", opacity: .55 }, svg);
    if (t > 0) {
      el("polyline", { points: path(r.past_a, r.pred_a), fill: "none", stroke: A, "stroke-width": 3, "stroke-linecap": "round" }, svg);
      el("polyline", { points: path(r.past_b, r.pred_b), fill: "none", stroke: B, "stroke-width": 3, "stroke-linecap": "round" }, svg);
      el("polyline", { points: path(r.past_a, r.cf_a), fill: "none", stroke: CF, "stroke-width": 3, "stroke-dasharray": "7 5", "stroke-linecap": "round" }, svg);
      if (reacts) el("polyline", { points: path(r.past_b, r.cf_b), fill: "none", stroke: RE, "stroke-width": 3, "stroke-dasharray": "7 5", "stroke-linecap": "round" }, svg);
    }
    // separation at time t
    if (t > 0) {
      const pa = r.pred_a[t - 1], pb = r.pred_b[t - 1], ca = r.cf_a[t - 1], cb = r.cf_b[t - 1];
      const line = (p, q, color, dash) => el("line", { x1: px(p)[0], y1: px(p)[1], x2: px(q)[0], y2: px(q)[1], stroke: color, "stroke-width": 1.4, "stroke-dasharray": dash, opacity: .85 }, svg);
      line(pa, pb, css("--muted"), "");
      line(ca, cb, CF, "4 4");
      for (const [p, c, rad] of [[pa, A, 6], [pb, B, 6], [ca, CF, 6]]) el("circle", { cx: px(p)[0], cy: px(p)[1], r: rad, fill: c, stroke: css("--panel"), "stroke-width": 2 }, svg);
      if (reacts) el("circle", { cx: px(cb)[0], cy: px(cb)[1], r: 6, fill: RE, stroke: css("--panel"), "stroke-width": 2 }, svg);
    }
    // current positions
    for (const [p, c, name] of [[nowA, A, "A"], [nowB, B, "B"]]) {
      const [x, y] = px(p);
      el("circle", { cx: x, cy: y, r: 9, fill: c, stroke: css("--panel"), "stroke-width": 2.5 }, svg);
      const tx = el("text", { x, y: y + 3.6, "text-anchor": "middle", fill: "#fff", "font-size": 10, "font-weight": 700 }, svg); tx.textContent = name;
    }
  }

  // pan / zoom
  function bindBev() {
    let drag = null;
    svg.addEventListener("pointerdown", (e) => { if (!S.view) return; drag = { x: e.clientX, y: e.clientY, ox: S.view.ox, oy: S.view.oy }; svg.setPointerCapture(e.pointerId); });
    svg.addEventListener("pointermove", (e) => { if (!drag) return; S.view.ox = drag.ox + e.clientX - drag.x; S.view.oy = drag.oy + e.clientY - drag.y; renderBev(); });
    const end = () => { drag = null; };
    svg.addEventListener("pointerup", end); svg.addEventListener("pointercancel", end);
    svg.addEventListener("wheel", (e) => {
      if (!S.view) return; e.preventDefault();
      const rect = svg.getBoundingClientRect(), mx = e.clientX - rect.left, my = e.clientY - rect.top;
      const k = Math.exp(-e.deltaY * 0.0015), v = S.view, ns = Math.min(Math.max(v.s * k, 2), 400), f = ns / v.s;
      v.ox = mx - (mx - v.ox) * f; v.oy = my - (my - v.oy) * f; v.s = ns; renderBev();
    }, { passive: false });
    $("fit").addEventListener("click", () => { if (S.result) { fitView(); renderBev(); } });
    new ResizeObserver(() => { if (S.result) { fitView(); renderBev(); } drawTicks(); }).observe(svg.parentElement);
  }

  // ---------- rendering: results ----------
  function chip(r) { return `<span class="chip ${r}">${r}</span>`; }
  function renderResults() {
    const r = S.result, order = S.meta.risk.order;
    if (!r) { $("risk-card").innerHTML = `<h2>Risk</h2><p class="muted">No analysis yet.</p>`; $("metrics").innerHTML = ""; $("gap").innerHTML = ""; $("expl").innerHTML = ""; $("warn").hidden = true; $("unc-sec").hidden = true; $("att-sec").hidden = true; return; }
    const o = r.original, c = r.counterfactual, jump = order.indexOf(c.risk) - order.indexOf(o.risk);
    const verdict = jump > 0 ? `Risk rises ${jump} tier${jump > 1 ? "s" : ""} under this intervention.` : jump < 0 ? `Risk falls ${-jump} tier${jump < -1 ? "s" : ""} under this intervention.` : "Risk tier unchanged by this intervention.";
    const kindLabel = S.meta.interventions.find((i) => i.kind === r.intervention[0]).label;
    $("risk-card").innerHTML = `<h2>Predicted pair risk · A #${r.track_a}, B #${r.track_b}</h2><div class="risk-flow">${chip(o.risk)}<span class="arrow" aria-hidden="true">→</span>${chip(c.risk)}</div><p class="verdict">${verdict} <span class="muted">(${kindLabel.toLowerCase()} = ${r.intervention[1]})</span></p>`;
    const w = $("warn");
    w.hidden = r.plausible;
    if (!r.plausible) w.textContent = `The intervened history needs ${fmt(r.max_accel_cf_mps2)} m/s² (above the ${S.meta.risk.max_accel_mps2} m/s² limit): it is unphysical, so treat the counterfactual prediction with caution.`;
    const os = o.scene, cs = c.scene;
    const delta = (a, b, lowerIsWorse, d, unit) => {
      if (a === null && b === null) return `<span class="muted">–</span>`;
      if (a === null || b === null) return `<span class="delta ${b === null ? "better" : "worse"}">${b === null ? "no longer closing" : "now closing"}</span>`;
      const x = b - a; if (Math.abs(x) < 1e-9) return `<span class="muted">0</span>`;
      const worse = lowerIsWorse ? x < 0 : x > 0;
      return `<span class="delta ${worse ? "worse" : "better"}">${x > 0 ? "+" : "−"}${Math.abs(x).toFixed(d)}${unit}</span>`;
    };
    $("metrics").innerHTML = `<tr><th></th><th>Original</th><th>Counterfactual</th><th>Δ</th></tr>
      <tr><td>Closest approach</td><td>${fmt(os.min_distance_m, 2)} m</td><td>${fmt(cs.min_distance_m, 2)} m</td><td>${delta(os.min_distance_m, cs.min_distance_m, true, 2, " m")}</td></tr>
      <tr><td>Time to collision</td><td>${fmt(os.ttc_s)} s</td><td>${fmt(cs.ttc_s)} s</td><td>${delta(os.ttc_s, cs.ttc_s, true, 1, " s")}</td></tr>
      <tr><td>Closing speed</td><td>${(os.closing_speed_mps >= 0 ? "+" : "") + fmt(os.closing_speed_mps, 2)} m/s</td><td>${(cs.closing_speed_mps >= 0 ? "+" : "") + fmt(cs.closing_speed_mps, 2)} m/s</td><td>${delta(os.closing_speed_mps, cs.closing_speed_mps, true, 2, " m/s")}</td></tr>`;
    $("expl").innerHTML = `<span>${o.explanation}</span><span>${c.explanation}</span>`;
    renderUnc(r); renderAtt(r); renderGap();
  }
  function bar(label, pct, cls, text) {
    return `<div class="bar"><span>${label}</span><div class="track"><div class="fill ${cls}" style="width:${Math.max(0, Math.min(100, pct))}%"></div></div><span class="val mono">${text ?? pct.toFixed(0) + "%"}</span></div>`;
  }
  function renderUnc(r) {
    const sec = $("unc-sec"); sec.hidden = !r.uncertainty;
    if (!r.uncertainty) return;
    const o = r.uncertainty.original, c = r.uncertainty.counterfactual;
    $("unc").innerHTML = `<p class="hint">Share of ${o.n_samples} sampled futures that end in:</p>` +
      bar("Overlap · orig", o.p_overlap * 100, "") + bar("Overlap · cf", c.p_overlap * 100, "cf") +
      bar("MEDIUM+ · orig", o.p_flagged * 100, "") + bar("MEDIUM+ · cf", c.p_flagged * 100, "cf") +
      `<p class="hint">Median TTC: ${fmt(o.ttc_median_s)} s → ${fmt(c.ttc_median_s)} s</p>`;
  }
  function renderAtt(r) {
    const sec = $("att-sec"); sec.hidden = !r.attention_a || !r.attention_a.length;
    if (sec.hidden) return;
    const max = Math.max(...r.attention_a.map((a) => a.weight), 1e-9);
    $("att").innerHTML = r.attention_a.slice().sort((x, y) => y.weight - x.weight)
      .map((a) => bar(`#${a.track}`, (a.weight / max) * 100, "", `${(a.weight * 100).toFixed(1)}%`)).join("");
  }
  function renderGap() {
    const g = $("gap"); g.innerHTML = "";
    const r = S.result; if (!r) return;
    const W = g.clientWidth || 300, H = 150, m = { l: 34, r: 8, t: 8, b: 22 };
    const F = r.gap.length, fps = S.meta.fps, tx = (i) => m.l + ((i + 1) / F) * (W - m.l - m.r);
    const ymax = Math.max(6, ...r.gap, ...r.cf_gap) * 1.3, ty = (v) => H - m.b - (v / ymax) * (H - m.t - m.b);
    const L = (x1, y1, x2, y2, attrs) => el("line", { x1, y1, x2, y2, ...attrs }, g);
    L(m.l, H - m.b, W - m.r, H - m.b, { stroke: css("--line") }); L(m.l, m.t, m.l, H - m.b, { stroke: css("--line") });
    const rad = S.meta.risk.collision_radius_m;
    L(m.l, ty(rad), W - m.r, ty(rad), { stroke: css("--critical"), "stroke-dasharray": "4 4", opacity: .7 });
    el("text", { x: W - m.r, y: ty(rad) - 3, "text-anchor": "end", fill: css("--critical"), "font-size": 10 }, g).textContent = `overlap < ${rad} m`;
    for (const v of [0, Math.round(ymax / 2), Math.round(ymax)]) el("text", { x: m.l - 5, y: ty(v) + 3, "text-anchor": "end", fill: css("--muted"), "font-size": 10 }, g).textContent = v;
    for (const s of [0, Math.floor(F / fps / 2) || 1, F / fps]) if (s <= F / fps) el("text", { x: tx(s * fps - 1), y: H - 6, "text-anchor": "middle", fill: css("--muted"), "font-size": 10 }, g).textContent = `${+s.toFixed(1)}s`;
    const line = (vals, color, dash) => el("polyline", { points: vals.map((v, i) => `${tx(i).toFixed(1)},${ty(v).toFixed(1)}`).join(" "), fill: "none", stroke: color, "stroke-width": 2.2, "stroke-dasharray": dash, "stroke-linejoin": "round" }, g);
    line(r.gap, css("--text"), ""); line(r.cf_gap, css("--cf"), "6 4");
    const t = Math.min(S.t, F);
    if (t > 0) L(tx(t - 1), m.t, tx(t - 1), H - m.b, { stroke: css("--accent"), "stroke-width": 1.5, opacity: .8 });
    el("text", { x: m.l + 6, y: m.t + 10, fill: css("--text"), "font-size": 10 }, g).textContent = "— original";
    el("text", { x: m.l + 70, y: m.t + 10, fill: css("--cf"), "font-size": 10 }, g).textContent = "– – counterfactual";
  }
  function showEmpty(msg) { const e = $("bev-empty"); e.textContent = msg; e.hidden = false; svg.innerHTML = ""; }
  function hideEmpty() { $("bev-empty").hidden = true; }
  function renderAll() { renderBev(); renderResults(); renderTime(); }
  function renderTime() { const F = model() ? model().future_len : 0; $("t-label").textContent = `+${(S.t / S.meta.fps).toFixed(1)} s`; $("t").value = S.t; $("play-ico").innerHTML = S.playing ? `<path d="M7 5h4v14H7zM13 5h4v14h-4z" fill="currentColor"/>` : `<path d="M8 5v14l11-7z" fill="currentColor"/>`; $("play").setAttribute("aria-label", S.playing ? "Pause" : "Play prediction"); void F; }

  // ---------- events ----------
  function play() {
    S.playing = !S.playing;
    cancelAnimationFrame(S.raf);
    if (S.playing) {
      const F = model().future_len; if (S.t >= F) S.t = 0;
      let last = performance.now();
      const step = (now) => {
        if (!S.playing) return;
        if (now - last > 70) { last = now; S.t += 1; if (S.t >= F) { S.t = F; S.playing = false; } renderBev(); renderGap(); renderTime(); }
        if (S.playing) S.raf = requestAnimationFrame(step);
      };
      S.raf = requestAnimationFrame(step);
    }
    renderTime();
  }
  function bind() {
    bindBev();
    $("model").addEventListener("change", (e) => { S.model = e.target.value; S.result = null; S.pair = null; loadFrames(S.start); });
    $("seq").addEventListener("change", (e) => { S.seq = Number(e.target.value); S.result = null; S.pair = null; S.start = null; loadFrames(null); });
    $("frame").addEventListener("input", (e) => { const f = S.frames[Number(e.target.value)]; $("frame-label").textContent = `frame ${f.now}`; $("agents-label").textContent = `${f.agents} agents`; clearTimeout(bind.h); bind.h = setTimeout(() => loadScene(f.start).catch((er) => toast(er.message)), 120); });
    $("value").addEventListener("input", (e) => { S.value = Number(e.target.value); syncValueOut(); scheduleAnalyze(); });
    $("mask").addEventListener("change", (e) => { S.mask = e.target.checked; saveHash(); runAnalyze(); });
    $("samples").addEventListener("change", (e) => { S.samples = e.target.checked; saveHash(); runAnalyze(); });
    $("show-truth").addEventListener("change", renderBev); $("show-others").addEventListener("change", renderBev);
    $("t").addEventListener("input", (e) => { S.playing = false; S.t = Number(e.target.value); renderBev(); renderGap(); renderTime(); });
    $("play").addEventListener("click", play);
    $("theme").addEventListener("click", () => {
      const dark = document.documentElement.dataset.theme === "dark" || (!document.documentElement.dataset.theme && matchMedia("(prefers-color-scheme: dark)").matches);
      const next = dark ? "light" : "dark"; document.documentElement.dataset.theme = next;
      try { localStorage.setItem("theme", next); } catch (_) { /* ignore */ }
      drawTicks(); renderAll();
    });
    $("escalate").addEventListener("click", async () => {
      S.escalating = true; busy(true);
      try {
        const r = await api("/api/escalation", { model: S.model, seq: S.seq, kind: S.kind, value: S.value, mask: S.mask, samples: S.samples ? 30 : 0 });
        if (!r.found) { toast("This intervention does not raise the risk of any pair in this sequence."); return; }
        const res = r.result, idx = S.frames.findIndex((f) => f.start === res.start);
        if (idx >= 0) { $("frame").value = idx; await loadScene(res.start, res.track_a, res.track_b); }
        S.pair = { a: res.track_a, b: res.track_b }; renderPairs(); setResult(res);
      } catch (e) { toast(e.message); } finally { S.escalating = false; busy(false); }
    });
    window.addEventListener("keydown", (e) => { if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return; if (e.key === " ") { e.preventDefault(); play(); } });
  }
  init();
})();
