import torch
import numpy as np
import matplotlib.pyplot as plt

from model import TrajectoryPredictor
from counterfactual.simulate import increase_speed

model = TrajectoryPredictor()
model.load_state_dict(torch.load("model.pth"))
model.eval()

from utility.kitti_parser import parse_kitti_labels
from utility.trajectory_builder import build_trajectories
from utility.trajectory_builder import clean_trajectories
from dataset import create_sequences, normalize
 
label_path = r"E:/Coding/PROGRAMS/Deep learning Projects/CSIE/data/training/label_02/0000.txt"

labels = parse_kitti_labels(label_path)
trajectories = build_trajectories(labels)
cleaned = clean_trajectories(trajectories)

X, Y = create_sequences(cleaned)
X, Y = normalize(X, Y)

sample = X[10] 

inp = torch.tensor(sample, dtype=torch.float32).unsqueeze(0)
orig = model(inp).detach().numpy()[0]

cf_sample = increase_speed(sample, 1.5)
cf_inp = torch.tensor(cf_sample, dtype=torch.float32).unsqueeze(0)
cf = model(cf_inp).detach().numpy()[0]

past = sample
plt.plot(orig[:,0], orig[:,1], 'go-', label="Original Future")
plt.plot(cf[:,0], cf[:,1], 'ro-', label="Counterfactual")

for i in range(len(orig)):
    plt.arrow(orig[i,0], orig[i,1],
              cf[i,0]-orig[i,0],
              cf[i,1]-orig[i,1],
              head_width=0.002, color='black')

plt.plot(past[:,0], past[:,1], 'bo-', label="Past")


plt.legend()
plt.title("Counterfactual Trajectory")
plt.show()