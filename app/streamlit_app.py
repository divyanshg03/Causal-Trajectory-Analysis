"""Interactive counterfactual explorer.  Run:  streamlit run app/streamlit_app.py"""

from pathlib import Path

import numpy as np
import streamlit as st

from cftraj.analyze import analyze_pair, figure_result
from cftraj.counterfactual import INTERVENTIONS
from cftraj.data import group_indices, load_windows
from cftraj.model import load_model
from cftraj.pairs import all_pairs

st.set_page_config(page_title="Counterfactual trajectory explorer", layout="wide")
st.title("Counterfactual scene reasoning on KITTI")
st.caption(
    "Pick two agents, rewrite one agent's observed motion, and see how the predicted "
    "futures and collision risk change. Positions are bird's-eye-view meters."
)

with st.sidebar:
    label_dir = st.text_input("KITTI label_02 directory", "data/training/label_02")
    ckpts = sorted(str(p) for p in Path(".").glob("checkpoints/*.pt")) or ["checkpoints/social_h30.pt"]
    ckpt = st.selectbox("Model checkpoint", ckpts)
    seq = st.selectbox("Sequence", list(range(21)), index=18)


@st.cache_resource
def get_model(path):
    return load_model(path)


@st.cache_resource
def get_windows(directory, seq_id, past_len, future_len):
    return load_windows(Path(directory) / f"{seq_id:04d}.txt", past_len, future_len, seq=seq_id)


try:
    model = get_model(ckpt)
    windows = get_windows(label_dir, seq, model.past_len, model.future_len)
except (FileNotFoundError, ValueError) as err:
    st.error(str(err))
    st.stop()

keys = windows.group_keys()
groups = {int(windows.starts[g[0]]): g for g in group_indices(keys) if len(g) >= 2}
if not groups:
    st.warning("No frame in this sequence has two co-occurring agents with a full window.")
    st.stop()

with st.sidebar:
    frames = sorted(groups)
    start = st.select_slider("Window start frame", options=frames, value=frames[len(frames) // 2])
    st.caption(f"'Now' is frame {start + model.past_len - 1}.")

group = groups[start]
sub = type("S", (), {"X": windows.X[group], "group_keys": lambda self: keys[group]})()
pairs = [(int(group[a]), int(group[b])) for a, b in all_pairs(sub, 40.0)]
if not pairs:
    st.warning("No pair of agents within 40 m in this frame; try another frame.")
    st.stop()


def label(pair):
    i, j = pair
    d = np.linalg.norm(windows.X[i, -1] - windows.X[j, -1])
    return f"track {windows.tids[i]} (A) & {windows.tids[j]} (B) — {d:.1f} m apart"


with st.sidebar:
    pair = st.selectbox("Agent pair", pairs, format_func=label)
    kind = st.radio("Intervention on A's history", INTERVENTIONS, horizontal=True,
                    format_func={"speed": "Scale speed", "lateral": "Drift sideways"}.get)
    if kind == "speed":
        value = st.slider("Speed factor", 0.0, 3.0, 2.0, 0.1)
    else:
        value = st.slider("Lateral offset (m)", -6.0, 6.0, 2.0, 0.5)
    mask = False
    if model.uses_neighbors:
        mask = st.checkbox("Ignore scene context (mask neighbors)", value=False)

res = analyze_pair(model, windows, kind, value, pair=pair, mask_neighbors=mask)
o, c = res["original"]["scene"], res["counterfactual"]["scene"]

left, right = st.columns([3, 2])
with left:
    st.pyplot(figure_result(res))
with right:
    st.subheader(f"Risk: {o['risk']} → {c['risk']}")
    st.table({
        "": ["Closest approach (m)", "TTC (s)", "Closing speed (m/s)"],
        "Original": [f"{o['min_distance_m']:.2f}", f"{o['ttc_s']:.1f}",
                     f"{o['closing_speed_mps']:+.2f}"],
        "Counterfactual": [f"{c['min_distance_m']:.2f}", f"{c['ttc_s']:.1f}",
                           f"{c['closing_speed_mps']:+.2f}"],
    })
    st.write(res["original"]["explanation"])
    st.write(res["counterfactual"]["explanation"])
    if res["attention_a"] is not None:
        st.caption("Attention of agent A over its neighbors (track id: weight)")
        i = res["indices"][0]
        att = {int(t): round(float(w), 3)
               for t, w, m in zip(windows.ntids[i], res["attention_a"], windows.nmask[i]) if m}
        st.json(att)
st.caption(
    "The intervention rewrites A's observed past while keeping its current position; "
    "it is a what-if perturbation of the model input, not a causal model."
)
