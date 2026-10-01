import torch
from torch import nn


class TrajectoryTransformer(nn.Module):
    """Transformer encoder mapping ``past`` (B, T, 2) to ``future`` (B, F, 2)."""

    def __init__(
        self, input_dim=2, d_model=64, nhead=4, num_layers=2, future_steps=3, max_len=10
    ):
        super().__init__()
        self.future_steps = future_steps
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, future_steps * 2)

    def forward(self, x):
        if x.size(1) > self.pos_embedding.size(1):
            raise ValueError(
                f"Sequence length {x.size(1)} exceeds max_len={self.pos_embedding.size(1)}"
            )
        x = self.input_proj(x) + self.pos_embedding[:, : x.size(1), :]
        out = self.transformer(x)[:, -1, :]
        return self.fc(out).view(-1, self.future_steps, 2)


def load_model(path, device="cpu"):
    model = TrajectoryTransformer()
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device).eval()
