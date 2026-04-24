import torch
import numpy as np
import matplotlib.pyplot as plt

from utils.kitti_parser import parse_kitti_labels
from utils.trajectory_builder import build_trajectories
from utils.trajectory_builder import clean_trajectories
from dataset import create_sequences, normalize
from model import TrajectoryTransformer
from counterfactual.simulate import increase_speed
from scene_graph.graph import build_scene_graph


# ==============================
# 🔹 Helper Functions
# ==============================

def trajectories_intersect(traj1, traj2, thresh=0.05):
    for p1 in traj1:
        for p2 in traj2:
            if np.linalg.norm(p1 - p2) < thresh:
                return True
    return False


def find_interacting_pair(X):
    for i in range(len(X)):
        for j in range(i + 1, len(X)):
            if trajectories_intersect(X[i], X[j]):
                return X[i], X[j]

    print("⚠️ No strong interaction found, using fallback pair")
    return X[0], X[1]


def compute_min_distance(traj1, traj2):
    return min(np.linalg.norm(traj1[i] - traj2[i])
               for i in range(min(len(traj1), len(traj2))))


def relative_motion_towards(traj1, traj2):
    pos_diff = traj2[-1] - traj1[-1]

    vel1 = traj1[-1] - traj1[-2]
    vel2 = traj2[-1] - traj2[-2]
    vel_diff = vel2 - vel1

    norm = np.linalg.norm(pos_diff) + 1e-6
    return np.dot(pos_diff, vel_diff) / norm

def compute_ttc(distance, motion):
    if motion >= 0:
        return float('inf')  # not approaching

    return distance / abs(motion)
def classify_risk(distance, motion):
    ttc = compute_ttc(distance, motion)

    if ttc < 1:
        return "🔥 CRITICAL (collision imminent)"
    elif ttc < 3:
        return "🔥 HIGH (collision soon)"
    elif ttc < 5:
        return "⚠️ MEDIUM (potential risk)"
    elif distance < 0.02:
        return "⚠️ LOW (monitor)"
    else:
        return "✅ SAFE"
def velocity(traj):
    return traj[-1] - traj[-2]

def explain(scene):
    motion = scene["motion"]
    distance = scene["distance"]

    if motion < 0:
        ttc = distance / (abs(motion) + 1e-6)
        return (
            f"{scene['object_A']} is approaching {scene['object_B']} "
            f"(distance={distance:.4f}, TTC={ttc:.2f}s). "
            f"Risk: {scene['risk']}."
        )
    else:
        return (
            f"{scene['object_A']} is moving away from {scene['object_B']} "
            f"(distance={distance:.4f}). Risk: {scene['risk']}."
        )
# ==============================
# 🔹 Load Data
# ==============================

label_path = "E:/Coding/PROGRAMS/Deep learning Projects/CSIE/data/training/label_02/0000.txt"

labels = parse_kitti_labels(label_path)
trajectories = build_trajectories(labels)
cleaned = clean_trajectories(trajectories)

X, Y = create_sequences(cleaned)
X, Y = normalize(X, Y)

print("X shape:", X.shape)


# ==============================
# 🔹 Load Transformer Model
# ==============================

model = TrajectoryTransformer()
model.load_state_dict(torch.load("model.pth"))
model.eval()


# ==============================
# 🔹 Select interacting objects
# ==============================

sample1, sample2 = find_interacting_pair(X)


# ==============================
# 🔹 Predict Futures
# ==============================

inp1 = torch.tensor(sample1, dtype=torch.float32).unsqueeze(0)
inp2 = torch.tensor(sample2, dtype=torch.float32).unsqueeze(0)

pred1 = model(inp1).detach().numpy()[0]
pred2 = model(inp2).detach().numpy()[0]


# ==============================
# 🔹 Counterfactual
# ==============================

cf_sample1 = increase_speed(sample1, factor=2.0)

cf_pred1 = model(
    torch.tensor(cf_sample1, dtype=torch.float32).unsqueeze(0)
).detach().numpy()[0]


# ==============================
# 🔹 Scene Graph
# ==============================

scene_original = build_scene_graph(pred1, pred2)
scene_counterfactual = build_scene_graph(cf_pred1, pred2)


# ==============================
# 🔹 Metrics
# ==============================

dist_original = compute_min_distance(pred1, pred2)
dist_counterfactual = compute_min_distance(cf_pred1, pred2)

motion_original = relative_motion_towards(pred1, pred2)
motion_counterfactual = relative_motion_towards(cf_pred1, pred2)

print("\n--- Metrics ---")
print(f"Original Distance: {dist_original:.4f}")
print(f"Counterfactual Distance: {dist_counterfactual:.4f}")
print(f"Motion Score (Original): {motion_original:.6f}")
print(f"Motion Score (Counterfactual): {motion_counterfactual:.6f}")


# ==============================
# 🔹 Scene Graph Output
# ==============================

print("\n--- Scene Graph (Original) ---")
print(scene_original)

print("\n--- Scene Graph (Counterfactual) ---")
print(scene_counterfactual)


# ==============================
# 🔹 Risk Analysis
# ==============================

print("\n--- Risk Analysis ---")
print("Original:", classify_risk(dist_original, motion_original))
print("Counterfactual:", classify_risk(dist_counterfactual, motion_counterfactual))


# ==============================
# 🔹 Explanation
# ==============================

print("\n--- Explanation ---")
print(explain(scene_original))
print(explain(scene_counterfactual))


# ==============================
# 🔹 Visualization
# ==============================

plt.figure(figsize=(6, 6))

plt.plot(pred1[:, 0], pred1[:, 1], 'go-', label="Object A (Future)")
plt.plot(pred2[:, 0], pred2[:, 1], 'bo-', label="Object B (Future)")
plt.plot(cf_pred1[:, 0], cf_pred1[:, 1], 'ro-', label="Object A (Counterfactual)")

for i in range(len(pred1)):
    plt.arrow(pred1[i, 0], pred1[i, 1],
              cf_pred1[i, 0] - pred1[i, 0],
              cf_pred1[i, 1] - pred1[i, 1],
              head_width=0.002, color='black')

plt.legend()
plt.title("Transformer-based Counterfactual Collision Reasoning")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid()
plt.show()