import numpy as np

# def generate_data(n_points=200, noise=0.1):
#     X = np.random.randn(n_points, 2)
#     y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0)
#     X += np.random.uniform(-noise, noise, X.shape)
#     y = y.astype(int)
#     return X, y

def generate_data(n_points=200, noise=0.1):
    points = n_points // 4
    radius = 0.5
    
    centers = np.array([
        [-1, 1],  # верхний левый угол
        [1, 1],   # верхний правый угол
        [-1, -1], # нижний левый угол
        [1, -1]   # нижний правый угол
    ])
    
    X = np.vstack([
        np.random.randn(points, 2) * radius + center for center in centers
    ])
    
    y = np.array([1] * points + [0] * points + 
                 [0] * points + [1] * points)
    
    return X, y
