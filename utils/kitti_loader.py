import os
import cv2
import numpy as np

def load_kitti_sequence(image_folder):
    images = sorted(os.listdir(image_folder))
    frames = []

    for img_name in images:
        img_path = os.path.join(image_folder, img_name)
        frame = cv2.imread(img_path)
        frames.append(frame)

    return frames


if __name__ == "__main__":
    frames = load_kitti_sequence(r"E:\Coding\PROGRAMS\Deep learning Projects\CSIE\data\training\image_02\0000")
    print(f"Loaded {len(frames)} frames")