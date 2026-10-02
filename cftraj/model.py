"""Trajectory predictors.

All models take the absolute past ``(B, P, 2)`` in meters (and, for the social
model, the pasts of neighboring agents) and return the absolute future
``(B, F, 2)``. They internally re-center on the last observed point
(translation invariance) and predict displacements from it, optionally as a
correction on top of constant-velocity extrapolation.
"""

import numpy as np
import torch
from torch import nn

SCALE = 1.0  # meters; input/output normalization constant
OFFSET_SCALE = 10.0  # meters; normalization of neighbor offsets


class _Base(nn.Module):
    uses_neighbors = False

    def __init__(self, past_len, future_len, scale=SCALE, residual=True, cv_k=3,
                 linear_skip=False, n_modes=1):
        super().__init__()
        if n_modes < 1:
            raise ValueError("n_modes must be at least 1")
        self.n_modes = n_modes
        if past_len < 2:
            raise ValueError("past_len must be at least 2 to estimate velocity")
        self.past_len, self.future_len, self.scale = past_len, future_len, scale
        self.residual, self.cv_k = residual, min(cv_k, past_len - 1)
        # optional linear path on the relative past (e.g. initialised from ridge regression)
        self.skip = nn.Linear(past_len * 2, future_len * 2) if linear_skip else None
        self.mode_logits = None  # (B, K) after a forward pass of a multi-modal model
        if self.skip is not None:
            nn.init.zeros_(self.skip.weight)
            nn.init.zeros_(self.skip.bias)

    def init_skip(self, weight):
        """Set the linear path from a ridge solution ``(2P, 2F)`` in relative coordinates."""
        with torch.no_grad():
            self.skip.weight.copy_(torch.as_tensor(weight.T, dtype=self.skip.weight.dtype))
            self.skip.bias.zero_()

    def _encode(self, rel, ctx):  # -> (B, K, F, 2) in normalized units, (B, K) logits
        raise NotImplementedError

    def forward_modes(self, past, nbr=None, nmask=None):
        """All ``n_modes`` futures ``(B, K, F, 2)`` and their logits ``(B, K)``."""
        if past.size(1) != self.past_len:
            raise ValueError(f"Expected {self.past_len} past steps, got {past.size(1)}")
        last = past[:, -1:, :]
        rel = (past - last) / self.scale

        ctx = None
        if self.uses_neighbors and nbr is not None:
            nlast = nbr[:, :, -1:, :]
            ctx = ((nbr - nlast) / self.scale,  # neighbor motion, own frame
                   (nlast[:, :, 0, :] - last) / OFFSET_SCALE,  # where it is vs the target
                   nmask)

        out, logits = self._encode(rel, ctx)  # (B, K, F, 2), (B, K)
        if self.skip is not None:
            out = out + self.skip(rel.flatten(1)).view(-1, 1, self.future_len, 2)
        if self.residual:  # learn a correction on top of constant-velocity extrapolation
            vel = (rel[:, -1] - rel[:, -1 - self.cv_k]) / self.cv_k
            steps = torch.arange(1, self.future_len + 1, device=past.device, dtype=past.dtype)
            out = out + steps[None, None, :, None] * vel[:, None, None, :]
        self.mode_logits = logits
        return last[:, None] + out * self.scale, logits

    def forward(self, past, nbr=None, nmask=None):
        """Single best future ``(B, F, 2)`` (the most likely mode)."""
        modes, logits = self.forward_modes(past, nbr, nmask)
        if self.n_modes == 1:
            return modes[:, 0]
        best = logits.argmax(dim=1)
        return modes[torch.arange(len(modes), device=modes.device), best]

    def _head(self, h, fc, mode_fc):
        """Shared output head: ``fc`` -> K futures, ``mode_fc`` -> K logits (or zeros)."""
        out = fc(h).view(-1, self.n_modes, self.future_len, 2)
        logits = mode_fc(h) if mode_fc is not None else h.new_zeros(h.size(0), 1)
        return out, logits


def _temporal_encoder(d_model, nhead, num_layers, dim_ff, dropout):
    layer = nn.TransformerEncoderLayer(
        d_model=d_model, nhead=nhead, dim_feedforward=dim_ff, dropout=dropout, batch_first=True
    )
    return nn.TransformerEncoder(layer, num_layers=num_layers)


class TrajectoryTransformer(_Base):
    def __init__(self, past_len=10, future_len=10, d_model=64, nhead=4, num_layers=2,
                 dim_ff=128, dropout=0.1, scale=SCALE, residual=True, linear_skip=False,
                 n_modes=1):
        super().__init__(past_len, future_len, scale, residual, linear_skip=linear_skip,
                         n_modes=n_modes)
        self.input_proj = nn.Linear(2, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, past_len, d_model) * 0.02)
        self.transformer = _temporal_encoder(d_model, nhead, num_layers, dim_ff, dropout)
        self.fc = nn.Linear(d_model, n_modes * future_len * 2)
        self.mode_fc = nn.Linear(d_model, n_modes) if n_modes > 1 else None

    def _encode(self, rel, ctx):
        h = self.transformer(self.input_proj(rel) + self.pos_embedding)[:, -1, :]
        return self._head(h, self.fc, self.mode_fc)


class TrajectoryLSTM(_Base):
    def __init__(self, past_len=10, future_len=10, hidden=64, num_layers=2, dropout=0.1,
                 scale=SCALE, residual=True, linear_skip=False, n_modes=1):
        super().__init__(past_len, future_len, scale, residual, linear_skip=linear_skip,
                         n_modes=n_modes)
        self.lstm = nn.LSTM(2, hidden, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden, n_modes * future_len * 2)
        self.mode_fc = nn.Linear(hidden, n_modes) if n_modes > 1 else None

    def _encode(self, rel, ctx):
        out, _ = self.lstm(rel)
        return self._head(out[:, -1, :], self.fc, self.mode_fc)


class TrajectorySocial(_Base):
    """Temporal Transformer + attention over neighboring agents.

    The target and each neighbor are encoded with a shared temporal encoder;
    the target then attends over its neighbors (keys carry the neighbor's
    position relative to the target). A learned null key lets the model ignore
    neighbors entirely. The attention over neighbors of the last forward pass
    is kept in ``last_attn`` (B, K) for interpretability.
    """

    uses_neighbors = True

    def __init__(self, past_len=10, future_len=30, d_model=64, nhead=4, num_layers=2,
                 dim_ff=128, dropout=0.1, scale=SCALE, residual=True, linear_skip=False,
                 n_modes=1):
        super().__init__(past_len, future_len, scale, residual, linear_skip=linear_skip,
                         n_modes=n_modes)
        self.input_proj = nn.Linear(2, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, past_len, d_model) * 0.02)
        self.transformer = _temporal_encoder(d_model, nhead, num_layers, dim_ff, dropout)
        self.offset_mlp = nn.Sequential(nn.Linear(2, d_model), nn.ReLU(),
                                        nn.Linear(d_model, d_model))
        self.null = nn.Parameter(torch.zeros(1, 1, d_model))
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.mix = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU())
        self.fc = nn.Linear(d_model, n_modes * future_len * 2)
        self.mode_fc = nn.Linear(d_model, n_modes) if n_modes > 1 else None
        self.last_attn = None

    def _temporal(self, rel):
        return self.transformer(self.input_proj(rel) + self.pos_embedding)[:, -1, :]

    def _encode(self, rel, ctx):
        h_t = self._temporal(rel)  # (B, d)
        keys = self.null.expand(h_t.size(0), 1, -1)
        pad = torch.zeros(h_t.size(0), 1, dtype=torch.bool, device=h_t.device)
        if ctx is not None:
            n_rel, n_off, nmask = ctx
            b, k = n_rel.shape[:2]
            h_n = self._temporal(n_rel.reshape(b * k, *n_rel.shape[2:])).view(b, k, -1)
            keys = torch.cat([keys, h_n + self.offset_mlp(n_off)], dim=1)
            pad = torch.cat([pad, ~nmask], dim=1)
        ctx_vec, w = self.attn(h_t[:, None, :], keys, keys, key_padding_mask=pad,
                               need_weights=True)
        self.last_attn = w[:, 0, 1:].detach()  # drop the null key
        h = self.norm(h_t + ctx_vec[:, 0])
        h = h + self.mix(h)
        return self._head(h, self.fc, self.mode_fc)


ARCHS = {
    "transformer": TrajectoryTransformer,
    "lstm": TrajectoryLSTM,
    "social": TrajectorySocial,
}


def build_model(arch, **config):
    if arch not in ARCHS:
        raise ValueError(f"Unknown arch '{arch}'. Choose from {sorted(ARCHS)}")
    return ARCHS[arch](**config)


def save_checkpoint(path, model, arch, config, extra=None):
    torch.save({"arch": arch, "config": config, "state_dict": model.state_dict(),
                **(extra or {})}, path)


def load_model(path, device="cpu"):
    """Load a checkpoint written by :func:`save_checkpoint`."""
    ckpt = torch.load(path, map_location=device)
    if not isinstance(ckpt, dict) or "config" not in ckpt:
        raise ValueError(
            f"{path} is a legacy checkpoint (pre-0.3). Retrain with 'python -m cftraj train'."
        )
    model = build_model(ckpt["arch"], **ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    model.arch = ckpt["arch"]
    return model.to(device).eval()


@torch.no_grad()
def predict_numpy(model, X, device="cpu", batch_size=2048, nbr=None, nmask=None):
    """Batched prediction on a numpy array (N, P, 2) -> (N, F, 2)."""
    model.eval()
    outs = []
    for i in range(0, len(X), batch_size):
        sl = slice(i, i + batch_size)
        xb = torch.as_tensor(X[sl], dtype=torch.float32, device=device)
        nb = nm = None
        if model.uses_neighbors and nbr is not None:
            nb = torch.as_tensor(nbr[sl], dtype=torch.float32, device=device)
            nm = torch.as_tensor(nmask[sl], dtype=torch.bool, device=device)
        outs.append(model(xb, nb, nm).cpu().numpy())
    return (
        np.concatenate(outs) if outs else np.zeros((0, model.future_len, 2), np.float32)
    )


@torch.no_grad()
def predict_modes_numpy(model, X, device="cpu", batch_size=2048, nbr=None, nmask=None):
    """All modes for a numpy batch: ``(N, K, F, 2)`` futures and ``(N, K)`` probabilities."""
    model.eval()
    futs, probs = [], []
    for i in range(0, len(X), batch_size):
        sl = slice(i, i + batch_size)
        xb = torch.as_tensor(X[sl], dtype=torch.float32, device=device)
        nb = nm = None
        if model.uses_neighbors and nbr is not None:
            nb = torch.as_tensor(nbr[sl], dtype=torch.float32, device=device)
            nm = torch.as_tensor(nmask[sl], dtype=torch.bool, device=device)
        modes, logits = model.forward_modes(xb, nb, nm)
        futs.append(modes.cpu().numpy())
        probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
    if not futs:
        return (np.zeros((0, model.n_modes, model.future_len, 2), np.float32),
                np.zeros((0, model.n_modes), np.float32))
    return np.concatenate(futs), np.concatenate(probs)


@torch.no_grad()
def sample_numpy(model, X, n_samples=20, device="cpu", nbr=None, nmask=None, seed=0):
    """Stochastic predictions ``(S, N, F, 2)`` for uncertainty estimates.

    Multi-modal models draw a mode per sample from the mode probabilities;
    single-mode models use Monte-Carlo dropout (dropout kept active).
    """
    gen = np.random.default_rng(seed)
    if model.n_modes > 1:
        futs, probs = predict_modes_numpy(model, X, device, nbr=nbr, nmask=nmask)
        cum = probs.cumsum(axis=1)
        out = []
        for _ in range(n_samples):
            pick = (gen.random((len(X), 1)) > cum).sum(axis=1).clip(max=model.n_modes - 1)
            out.append(futs[np.arange(len(X)), pick])
        return np.stack(out)
    torch.manual_seed(seed)
    was_training = model.training
    model.train()  # enable dropout
    try:
        xb = torch.as_tensor(X, dtype=torch.float32, device=device)
        nb = nm = None
        if model.uses_neighbors and nbr is not None:
            nb = torch.as_tensor(nbr, dtype=torch.float32, device=device)
            nm = torch.as_tensor(nmask, dtype=torch.bool, device=device)
        out = np.stack([model(xb, nb, nm).cpu().numpy() for _ in range(n_samples)])
    finally:
        model.train(was_training)
    return out


def predict_windows(model, w, device="cpu", mask_neighbors=False):
    """Predict for every window, using its neighbors if the model is social."""
    nmask = None if w.nmask is None else (np.zeros_like(w.nmask) if mask_neighbors else w.nmask)
    return predict_numpy(model, w.X, device, nbr=w.nbr, nmask=nmask)
