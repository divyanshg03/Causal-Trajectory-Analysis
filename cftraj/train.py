import torch
from torch import nn, optim

from .model import TrajectoryTransformer


def train_model(model, X, Y, epochs=250, lr=1e-3, out_path="model.pth", seed=0, log_every=10):
    """Full-batch Adam training on normalized windows; saves ``out_path``."""
    torch.manual_seed(seed)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    X = torch.as_tensor(X, dtype=torch.float32)
    Y = torch.as_tensor(Y, dtype=torch.float32)

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X), Y)
        loss.backward()
        optimizer.step()
        if epoch % log_every == 0 or epoch == epochs:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

    torch.save(model.state_dict(), out_path)
    return model


def run_training(windows, epochs, lr, out_path, seed):
    if len(windows) == 0:
        raise SystemExit("No training windows could be built from the given labels.")
    print(f"X shape: {windows.X.shape}, Y shape: {windows.Y.shape}")
    torch.manual_seed(seed)
    model = TrajectoryTransformer()
    train_model(model, windows.X, windows.Y, epochs, lr, out_path, seed)
    print(f"Saved weights to {out_path}")
