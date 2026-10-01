"""KITTI tracking label parsing, trajectory building and window creation."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

KITTI_WIDTH = 1242
KITTI_HEIGHT = 375


def parse_kitti_labels(label_file, ignore_types=("DontCare",)):
    """Parse a KITTI tracking label file into ``{frame: [object dicts]}``."""
    data = {}
    with open(label_file, "r") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 10:
                continue
            obj_type = parts[2]
            if obj_type in ignore_types:
                continue
            frame = int(parts[0])
            data.setdefault(frame, []).append(
                {
                    "track_id": int(parts[1]),
                    "type": obj_type,
                    "bbox": [float(v) for v in parts[6:10]],
                }
            )
    return data


def build_trajectories(label_data):
    """Convert per-frame labels into ``{track_id: [(frame, cx, cy), ...]}``."""
    trajectories = {}
    for frame, objects in label_data.items():
        for obj in objects:
            x1, y1, x2, y2 = obj["bbox"]
            trajectories.setdefault(obj["track_id"], []).append(
                (frame, (x1 + x2) / 2, (y1 + y2) / 2)
            )
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

    ``tids[i]`` and ``starts[i]`` give the track id and the first frame of
    window ``i``, so two windows describe co-occurring agents iff they share
    ``starts`` and differ in ``tids``.
    """

    X: np.ndarray  # (N, past_len, 2)
    Y: np.ndarray  # (N, future_len, 2)
    tids: np.ndarray  # (N,)
    starts: np.ndarray  # (N,)

    def __len__(self):
        return len(self.X)


def create_windows(trajectories, past_len=5, future_len=3):
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
    )


def normalize(arr, width=KITTI_WIDTH, height=KITTI_HEIGHT):
    """Scale pixel coordinates to ``[0, 1]``. Returns a new array."""
    out = np.array(arr, dtype=np.float32, copy=True)
    out[..., 0] /= width
    out[..., 1] /= height
    return out


def load_windows(label_path, past_len=5, future_len=3, min_length=8):
    """Label file -> normalized :class:`Windows` (labels file path -> model input)."""
    if not Path(label_path).is_file():
        raise FileNotFoundError(
            f"Label file not found: {label_path}. See 'Dataset setup' in the README."
        )
    traj = clean_trajectories(
        build_trajectories(parse_kitti_labels(label_path)), min_length=min_length
    )
    w = create_windows(traj, past_len, future_len)
    return Windows(normalize(w.X), normalize(w.Y), w.tids, w.starts)


def load_kitti_frames(image_folder):
    """Load every image in a KITTI image folder (requires opencv-python)."""
    import cv2

    folder = Path(image_folder)
    return [cv2.imread(str(p)) for p in sorted(folder.iterdir())]
