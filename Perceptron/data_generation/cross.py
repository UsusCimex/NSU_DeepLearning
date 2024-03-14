import numpy as np

def generate_data(n_points=200, noise=0.1):
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

if __name__ == "__main__":
    print(greet("Alice"))