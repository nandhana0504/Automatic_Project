import numpy as np

NUM_LANDMARKS = 19

def load_gt(anno_path):
    """
    Loads ground truth landmarks from annotation file
    """
    points = []
    with open(anno_path, "r") as f:
        lines = f.readlines()

    for i in range(NUM_LANDMARKS):
        x, y = lines[i].strip().split(",")
        points.append([float(x), float(y)])

    return np.array(points)
