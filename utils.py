import os

def load_csv(filename):
    data = []

    base_dir = os.path.dirname(__file__)
    file_path = os.path.join(base_dir, filename)

    with open(file_path) as f:
        next(f)
        for line in f:
            x, y, label = line.strip().split(",")
            data.append(([float(x), float(y)], int(label)))

    return data
