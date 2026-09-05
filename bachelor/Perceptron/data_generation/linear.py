import numpy as np

def generate_data(n_points=200, noise=0.1):
    X = np.random.randn(n_points, 2) * 2 - 1
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    X += np.random.uniform(-noise, noise, X.shape)
    return X, y