import numpy as np

def generate_xor_data(n_points=200, noise=0.1):
    rng = np.random.RandomState(42)
    X = rng.randn(n_points, 2)
    y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0)
    X += rng.uniform(-noise, noise, X.shape)
    y = y.astype(int)
    return X, y

def generate_linear_data(n_points=200, noise=0.1):
    rng = np.random.RandomState(42)
    X = rng.randn(n_points, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    X += rng.uniform(-noise, noise, X.shape)
    return X, y

def generate_spiral_data(n_points=100, noise=0.2):
    n = np.sqrt(np.random.rand(n_points, 1)) * 720 * (2 * np.pi) / 360
    d1x = -np.cos(n) * n + np.random.rand(n_points, 1) * noise
    d1y = np.sin(n) * n + np.random.rand(n_points, 1) * noise
    return (np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))), 
            np.hstack((np.zeros(n_points), np.ones(n_points))))

def generate_crossing_hooks_data(n_points=200, noise=0.1):
    t = np.linspace(-1, 1, n_points // 2)
    x_upper = t
    y_upper = np.abs(t) ** 0.5 - 0.5
    x_lower = t
    y_lower = -np.abs(t) ** 0.5 + 0.5

    x_upper += np.random.normal(scale=noise, size=x_upper.shape)
    y_upper += np.random.normal(scale=noise, size=y_upper.shape)
    x_lower += np.random.normal(scale=noise, size=x_lower.shape)
    y_lower += np.random.normal(scale=noise, size=y_lower.shape)

    X = np.vstack((np.column_stack((x_upper, y_upper)), np.column_stack((x_lower, y_lower))))
    y = np.array([1] * (n_points // 2) + [0] * (n_points // 2))
    return X, y

def get_dataset(dataset_choice, n_points=200, noise=0.1):
    if dataset_choice == "linear":
        return generate_linear_data(n_points, noise)
    elif dataset_choice == "spiral":
        return generate_spiral_data(n_points, noise)
    elif dataset_choice == "xor":
        return generate_xor_data(n_points, noise)
    elif dataset_choice == "cross":
        return generate_crossing_hooks_data(n_points, noise)
    else:
        raise ValueError("Invalid dataset choice. Please choose from 'linear', 'spiral', 'xor', 'cross'.")