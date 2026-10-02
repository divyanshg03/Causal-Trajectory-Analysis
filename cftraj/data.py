"""KITTI tracking labels -> metric bird's-eye-view trajectories and windows.

Agents are located by the KITTI 3D label ``location`` field. Camera
coordinates are ``x`` right, ``y`` down, ``z`` forward, so the ground-plane
(bird's-eye view) position is ``(x, z)`` in **meters**. Coordinates are in the
ego-camera frame of each frame (no OXTS ego-motion compensation), which keeps
*relative* quantities between two agents in the same frame (distance, closing
speed, TTC) physically meaningful.
"""

import hashlib
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional

import numpy as np

# Official KITTI tracking training sequences: 0000-0020 (21 sequences).
# Test = the last three. Validation sequences were chosen among the other 18 so
# that their constant-velocity error at a 3 s horizon matches the pooled
# average of those 18 (label-derived, test never consulted); the obvious
# split 15-17 is dominated by a nearly static sequence and is unrepresentative.
_VAL = (4, 11, 17)
SPLITS = {
    "train": tuple(s for s in range(18) if s not in _VAL),
    "val": _VAL,
    "test": (18, 19, 20),
}
KITTI_FPS = 10.0
N_NEIGHBORS = 8
NEIGHBOR_RADIUS = 30.0  # meters, measured between last observed points


def split_label(split):
    return ", ".join(f"{s:04d}" for s in SPLITS[split])


def parse_kitti_labels(label_file, ignore_types=("DontCare",)):
    """Parse a KITTI tracking label file into ``{frame: [object dicts]}``."""
    data = {}
    with open(label_file, "r") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 16:
                continue
            obj_type = parts[2]
            if obj_type in ignore_types:
                continue
            data.setdefault(int(parts[0]), []).append(
                {
                    "track_id": int(parts[1]),
                    "type": obj_type,
                    "bbox": [float(v) for v in parts[6:10]],
                    "dims": [float(v) for v in parts[10:13]],  # h, w, l
                    "location": [float(v) for v in parts[13:16]],  # x, y, z (camera)
                }
            )
    return data


def build_trajectories(label_data):
    """Convert per-frame labels into ``{track_id: [(frame, x, z), ...]}`` in meters."""
    trajectories = {}
    for frame, objects in label_data.items():
        for obj in objects:
            x, _, z = obj["location"]
            trajectories.setdefault(obj["track_id"], []).append((frame, x, z))
    return trajectories


def clean_trajectories(trajectories, min_length=8):
    """Drop short tracks and sort the remaining ones by frame."""
    return {
        tid: sorted(traj, key=lambda p: p[0])
        for tid, traj in trajectories.items()
        if len(traj) >= min_length
    }


@dataclass
class Windows:
    """Sliding windows plus the metadata needed to pair agents in time.

    Two windows describe co-occurring agents iff they share ``(seqs, starts)``
    and differ in ``tids``. Coordinates are absolute BEV meters.

    ``nbr`` / ``nmask`` / ``ntids`` hold, for every window, the observed pasts
    of its ``K`` nearest co-occurring agents (padded; ``nmask`` marks real
    ones). They are filled by :func:`attach_neighbors`.
    """

    X: np.ndarray  # (N, past_len, 2)
    Y: np.ndarray  # (N, future_len, 2)
    tids: np.ndarray  # (N,)
    starts: np.ndarray  # (N,)
    seqs: np.ndarray  # (N,)
    nbr: Optional[np.ndarray] = None  # (N, K, past_len, 2)
    nmask: Optional[np.ndarray] = None  # (N, K) bool
    ntids: Optional[np.ndarray] = None  # (N, K) int, -1 for padding

    def __len__(self):
        return len(self.X)

    @property
    def past_len(self):
        return self.X.shape[1]

    @property
    def future_len(self):
        return self.Y.shape[1]

    def group_keys(self):
        """One integer per (sequence, start frame): windows sharing it co-occur."""
        return self.seqs.astype(np.int64) * 10**7 + self.starts


def create_windows(trajectories, past_len=10, future_len=10, seq=0):
    """Slide a window over each track; only contiguous-frame windows are kept."""
    total = past_len + future_len
    X, Y, tids, starts = [], [], [], []

    for tid, traj in trajectories.items():
        frames = [p[0] for p in traj]
        coords = [(p[1], p[2]) for p in traj]

        for i in range(len(coords) - total + 1):
            if frames[i + total - 1] - frames[i] != total - 1:
                continue  # gap in the track
            X.append(coords[i : i + past_len])
            Y.append(coords[i + past_len : i + total])
            tids.append(tid)
            starts.append(frames[i])

    return Windows(
        X=np.array(X, dtype=np.float32).reshape(-1, past_len, 2),
        Y=np.array(Y, dtype=np.float32).reshape(-1, future_len, 2),
        tids=np.array(tids, dtype=np.int64),
        starts=np.array(starts, dtype=np.int64),
        seqs=np.full(len(X), seq, dtype=np.int64),
    )


def group_indices(keys):
    """Indices of each distinct key, via one argsort: ``[array, array, ...]``."""
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    cuts = np.flatnonzero(np.diff(sorted_keys)) + 1
    return np.split(order, cuts) if len(order) else []


def attach_neighbors(w, k=N_NEIGHBORS, radius=NEIGHBOR_RADIUS):
    """Fill ``nbr``, ``nmask`` and ``ntids`` with each window's nearest co-occurring agents."""
    n = len(w)
    w.nbr = np.zeros((n, k, w.past_len, 2), dtype=np.float32)
    w.nmask = np.zeros((n, k), dtype=bool)
    w.ntids = np.full((n, k), -1, dtype=np.int64)

    for idx in group_indices(w.group_keys()):
        if len(idx) < 2:
            continue
        last = w.X[idx, -1, :]
        d = np.linalg.norm(last[:, None, :] - last[None, :, :], axis=-1)
        np.fill_diagonal(d, np.inf)
        for a in range(len(idx)):
            order = np.argsort(d[a])[:k]
            order = order[d[a][order] <= radius]
            for slot, b in enumerate(order):
                w.nbr[idx[a], slot] = w.X[idx[b]]
                w.nmask[idx[a], slot] = True
                w.ntids[idx[a], slot] = w.tids[idx[b]]
    return w


def concat_windows(parts):
    parts = list(parts)
    if not parts:
        raise ValueError("No windows to concatenate")
    fields = {}
    for f in ("X", "Y", "tids", "starts", "seqs", "nbr", "nmask", "ntids"):
        arrs = [getattr(p, f) for p in parts]
        fields[f] = None if any(a is None for a in arrs) else np.concatenate(arrs)
    return Windows(**fields)


_WINDOW_FIELDS = tuple(f.name for f in fields(Windows))


def _cache_path(cache_dir, label_path, past_len, future_len, seq, neighbors):
    st = Path(label_path).stat()
    key = "|".join(map(str, (Path(label_path).resolve(), st.st_mtime_ns, st.st_size, past_len,
                             future_len, seq, neighbors, N_NEIGHBORS, NEIGHBOR_RADIUS)))
    return Path(cache_dir) / f"windows_{hashlib.sha1(key.encode()).hexdigest()[:16]}.npz"


def _save_windows(path, w):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{f: getattr(w, f) for f in _WINDOW_FIELDS if getattr(w, f) is not None})


def _load_cached(path):
    with np.load(path) as z:
        return Windows(**{f: (z[f] if f in z.files else None) for f in _WINDOW_FIELDS})


def load_windows(label_path, past_len=10, future_len=10, seq=0, neighbors=True, cache_dir=None):
    """One label file -> :class:`Windows` (with neighbors unless disabled).

    With ``cache_dir`` the (slow) neighbor search is cached as ``.npz``; the key covers
    the file's path, size and mtime plus every parameter, so edits invalidate it.
    """
    if not Path(label_path).is_file():
        raise FileNotFoundError(
            f"Label file not found: {label_path}. See 'Dataset setup' in the README."
        )
    cache = None
    if cache_dir is not None:
        cache = _cache_path(cache_dir, label_path, past_len, future_len, seq, neighbors)
        if cache.is_file():
            try:
                return _load_cached(cache)
            except (OSError, ValueError, KeyError):
                pass  # corrupt cache: rebuild
    traj = clean_trajectories(
        build_trajectories(parse_kitti_labels(label_path)), min_length=past_len + future_len
    )
    w = create_windows(traj, past_len, future_len, seq=seq)
    w = attach_neighbors(w) if neighbors else w
    if cache is not None:
        _save_windows(cache, w)
    return w


def load_split(label_dir, split, past_len=10, future_len=10, cache_dir=None):
    """All sequences of ``split`` ('train' | 'val' | 'test') -> :class:`Windows`."""
    return load_sequences(label_dir, SPLITS[split], past_len, future_len, cache_dir)


def load_sequences(label_dir, seqs, past_len=10, future_len=10, cache_dir=None):
    """Windows of the given KITTI sequence ids, concatenated."""
    parts = [
        load_windows(Path(label_dir) / f"{s:04d}.txt", past_len, future_len, seq=s,
                     cache_dir=cache_dir)
        for s in seqs
    ]
    return concat_windows(parts)


def kfold_sequences(k=5, seqs=None):
    """Deterministic sequence-level folds: ``[(train_seqs, test_seqs), ...]``.

    ``seqs`` defaults to all 21 KITTI tracking training sequences; folds are
    interleaved (every k-th sequence) so each mixes early and late recordings.
    """
    seqs = list(range(21)) if seqs is None else list(seqs)
    if not 2 <= k <= len(seqs):
        raise ValueError(f"k must be between 2 and {len(seqs)}")
    folds = [seqs[i::k] for i in range(k)]
    return [([s for s in seqs if s not in test], test) for test in folds]


def load_csv_trajectories(path):
    """Tracks from a generic CSV so other datasets can be plugged in.

    Columns (header required): ``seq,frame,track,x,y`` in meters (BEV). Returns
    ``{seq: {track_id: [(frame, x, y), ...]}}`` ready for
    :func:`clean_trajectories` / :func:`create_windows` (one call per sequence).
    """
    import csv

    out = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        missing = {"seq", "frame", "track", "x", "y"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        for row in reader:
            out.setdefault(int(row["seq"]), {}).setdefault(int(row["track"]), []).append(
                (int(row["frame"]), float(row["x"]), float(row["y"])))
    return out


def load_kitti_frames(image_folder):
    """Load every image in a KITTI image folder (requires opencv-python)."""
    import cv2

    folder = Path(image_folder)
    return [cv2.imread(str(p)) for p in sorted(folder.iterdir())]
