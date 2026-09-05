import numpy as np

def generate_data(n_points=200, noise=0.1):
    center1 = np.array([-1, 1])
    center2 = np.array([1, -1])
    radius = 0.5
    
    X1 = np.random.randn(n_points // 2, 2) * radius + center1
    X2 = np.random.randn(n_points // 2, 2) * radius + center2
    X = np.vstack((X1, X2))
    
    y = np.hstack((np.zeros(n_points // 2), np.ones(n_points // 2)))
    
    X += np.random.uniform(-noise, noise, X.shape)
    return X, y