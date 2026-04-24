import numpy as np

def create_sequences(trajectories, past_len=5, future_len=3):
    X, Y = [], []

    for tid, traj in trajectories.items():
        coords = [(x, y) for (_, x, y) in traj]

        if len(coords) < past_len + future_len:
            continue

        for i in range(len(coords) - past_len - future_len):
            past = coords[i:i+past_len]
            future = coords[i+past_len:i+past_len+future_len]

            X.append(past)
            Y.append(future)

    return np.array(X), np.array(Y)

def normalize(X, Y, width=1242, height=375):
    X[:, :, 0] /= width
    X[:, :, 1] /= height

    Y[:, :, 0] /= width
    Y[:, :, 1] /= height

    return X, Y