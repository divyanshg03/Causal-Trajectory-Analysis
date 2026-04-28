def parse_kitti_labels(label_file):
    data = {}

    with open(label_file, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()

        frame = int(parts[0])
        track_id = int(parts[1])
        obj_type = parts[2]

        bbox = list(map(float, parts[6:10]))

        if frame not in data:
            data[frame] = []

        data[frame].append({
            "track_id": track_id,
            "type": obj_type,
            "bbox": bbox
        })

    return data


if __name__ == "__main__":
    labels = parse_kitti_labels(r"E:\Coding\PROGRAMS\Deep learning Projects\CSIE\data\training\label_02\0000.txt")
    print(labels[0][:2])