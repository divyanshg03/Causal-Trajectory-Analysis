def build_trajectories(label_data):
    trajectories = {}

    for frame, objects in label_data.items():
        for obj in objects:
            tid = obj["track_id"]
            bbox = obj["bbox"]

            # center point
            x = (bbox[0] + bbox[2]) / 2
            y = (bbox[1] + bbox[3]) / 2

            if tid not in trajectories:
                trajectories[tid] = []

            trajectories[tid].append((frame, x, y))

    return trajectories

def clean_trajectories(trajectories, min_length=8):
    cleaned = {}

    for tid, traj in trajectories.items():
        if len(traj) >= min_length:
            traj = sorted(traj, key=lambda x: x[0])
            cleaned[tid] = traj

    return cleaned