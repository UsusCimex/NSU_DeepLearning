import numpy as np

def generate_data(n_points=200, noise=0.1):
    rng = np.random.RandomState(42)
    X = rng.randn(n_points, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    X += rng.uniform(-noise, noise, X.shape)
    return X, y