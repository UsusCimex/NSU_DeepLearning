import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

cm = LinearSegmentedColormap.from_list('blue_red', ['red', 'blue'], N=2)
def visualize_graph(model, X, y, iteration=None, loss=None):
    # Setting the range of values and the grid
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))

    # Predict over the grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Ensure contour levels are increasing and distinct
    levels = np.linspace(Z.min(), Z.max(), 3)
    if len(np.unique(levels)) < len(levels):
        if Z.max() == Z.min():
            levels = np.array([Z.min(), Z.min() + 1e-4, Z.min() + 2e-4])  # Add small distinct values
        else:
            levels = np.unique(levels)  # Remove duplicate levels

    # Plotting
    plt.contourf(xx, yy, Z, alpha=0.8, levels=levels, cmap=plt.cm.RdBu)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=cm, edgecolors='k')
    plt.title(f"Iteration {iteration}, Loss: {loss}")
    plt.show()

def visualize_loss(loss_history):
    if loss_history and len(loss_history) > 0:
        epochs = range(1, len(loss_history) + 1)  # Создаём список эпох (для оси X)
        plt.plot(epochs, loss_history, label='Training loss')
        plt.title('Model Loss Progression')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
    else:
        print("Loss history is empty or not provided.")

def plot_data(X, y):
    plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=cm, edgecolors='k')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Data points')
    plt.show()

def plot_lines(X, y, lines):
    plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=cm, edgecolors='k')
    for i, line in enumerate(lines):
        slope, intercept = line
        x_vals = np.array([min(X[:, 0]), max(X[:, 0])])
        y_vals = slope * x_vals + intercept
        plt.plot(x_vals, y_vals, label=f'Model {i+1}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Data points with Lines')
    plt.axis([-3, 3, -3, 3])
    plt.legend()
    plt.show()