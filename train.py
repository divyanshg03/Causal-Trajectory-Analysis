import torch
import torch.nn as nn
import torch.optim as optim

from utils.kitti_parser import parse_kitti_labels
from utils.trajectory_builder import build_trajectories
from utils.trajectory_builder import clean_trajectories
from dataset import create_sequences, normalize
from model import TrajectoryTransformer


def train_model(model, X, Y, epochs=250):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    X = torch.tensor(X, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32)

    for epoch in range(epochs):
        model.train()

        optimizer.zero_grad()
        output = model(X)

        loss = criterion(output, Y)
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), "model.pth")


if __name__ == "__main__":

    label_path = r"E:/Coding/PROGRAMS/Deep learning Projects/CSIE/data/training/label_02/0000.txt"

    labels = parse_kitti_labels(label_path)
    trajectories = build_trajectories(labels)

    cleaned = clean_trajectories(trajectories)

    X, Y = create_sequences(cleaned)

    print("X shape:", X.shape)
    print("Y shape:", Y.shape)

    if len(X) == 0:
        print("❌ No data to train on")
        exit()

    X, Y = normalize(X, Y)

    # 🔥 REAL transformer
    model = TrajectoryTransformer()

    train_model(model, X, Y, epochs=250)