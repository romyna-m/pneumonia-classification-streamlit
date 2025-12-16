import numpy as np

def load_file(path):
    return np.load(path).astype(np.float32)
