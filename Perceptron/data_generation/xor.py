import numpy as np

def generate_data(n_points=200, noise=0.1):
    rng = np.random.RandomState(42)
    X = rng.randn(n_points, 2)
    y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0)
    X += rng.uniform(-noise, noise, X.shape)
    y = y.astype(int)
    return X, y