import torch
import torch.nn as nn

class TrajectoryTransformer(nn.Module):
    def __init__(self, input_dim=2, d_model=64, nhead=4, num_layers=2, future_steps=3):
        super().__init__()

        self.future_steps = future_steps

        # input embedding
        self.input_proj = nn.Linear(input_dim, d_model)

        # positional encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, 10, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True
        )
     
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # output head
        self.fc = nn.Linear(d_model, future_steps * 2)

    def forward(self, x):
        # x: (batch, seq, 2)
        x = self.input_proj(x)

        # add positional encoding
        x = x + self.pos_embedding[:, :x.size(1), :]

        out = self.transformer(x)

        # use last token
        out = out[:, -1, :]

        out = self.fc(out)
        out = out.view(-1, self.future_steps, 2)

        return out