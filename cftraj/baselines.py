"""Classical baselines. Each takes X (N, P, 2) and returns (N, F, 2), in meters."""

import numpy as np


def stationary(X, future_len):
    """Agent stays at its last observed position."""
    return np.repeat(X[:, -1:, :], future_len, axis=1)


def constant_velocity(X, future_len, k=3):
    """Extrapolate the mean velocity over the last ``k`` steps."""
    k = min(k, X.shape[1] - 1)
    vel = (X[:, -1] - X[:, -1 - k]) / k
    steps = np.arange(1, future_len + 1, dtype=X.dtype)[None, :, None]
    return X[:, -1:, :] + steps * vel[:, None, :]


def kalman(X, future_len, q=0.05, r=0.5):
    """Constant-velocity Kalman filter over the past, then extrapolation.

    State ``[x, y, vx, vy]`` per frame. With identical noise settings and no
    missing data the gain is the same for every window, so it is computed once
    and the filter is applied to the whole batch.
    """
    dt = 1.0
    F = np.eye(4)
    F[0, 2] = F[1, 3] = dt
    H = np.zeros((2, 4))
    H[0, 0] = H[1, 1] = 1.0
    I2 = np.eye(2)
    Q = q * np.block([[0.25 * I2, 0.5 * I2], [0.5 * I2, I2]])
    R = r * I2

    P = r * np.diag([1.0, 1.0, 10.0, 10.0])  # scale-free: only q / r matters
    gains = []
    for _ in range(X.shape[1] - 1):
        P = F @ P @ F.T + Q
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        P = (np.eye(4) - K @ H) @ P
        gains.append(K)

    state = np.concatenate([X[:, 0], np.zeros_like(X[:, 0])], axis=1)  # (N, 4)
    for t, K in enumerate(gains, start=1):
        pred = state @ F.T
        state = pred + (X[:, t] - pred @ H.T) @ K.T

    out = []
    for _ in range(future_len):
        state = state @ F.T
        out.append(state[:, :2])
    return np.stack(out, axis=1).astype(X.dtype)


def tune_kalman(X, Y, qs=None):
    """Grid-search the process/measurement noise ratio ``q`` (with ``r = 1``).

    With the initial covariance scaled by ``r``, the Kalman gain depends only
    on ``q / r``, so a 1-D log-spaced search over ``q`` is exhaustive. Returns
    ``(q, r)``; a warning is printed if the optimum is on the grid edge.
    """
    qs = np.logspace(-3, 4, 29) if qs is None else np.asarray(qs)
    ades = [
        float(np.linalg.norm(kalman(X, Y.shape[1], q, 1.0) - Y, axis=-1).mean()) for q in qs
    ]
    best = int(np.argmin(ades))
    if best in (0, len(qs) - 1):
        print(f"warning: Kalman optimum q={qs[best]:g} is on the edge of the search grid")
    return float(qs[best]), 1.0


def fit_ridge(X, Y, lam=1e-1):
    """Closed-form ridge regression from relative past to relative future.

    Returns a predictor ``f(X, future_len) -> (N, F, 2)``. This is the best
    *linear* predictor and a strong sanity check for learned models.
    """
    def feat(a):
        return (a - a[:, -1:]).reshape(len(a), -1)

    A = feat(X)
    T = (Y - X[:, -1:]).reshape(len(X), -1)
    W = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ T)

    def predict(Xn, future_len=None):
        out = (feat(Xn) @ W).reshape(len(Xn), -1, 2)
        return Xn[:, -1:, :] + out.astype(Xn.dtype)

    predict.W = W
    return predict
