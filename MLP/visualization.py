import matplotlib.pyplot as plt
import numpy as np

def plot_decision_boundary(model, X, y, iteration, loss):
    """
    Функция для отображения границы решения и распределения точек датасета.
    :param model: Обученная модель нейронной сети
    :param X: Массив признаков
    :param y: Массив меток
    """
    # Задаем диапазон значений и сетку
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))

    # Прогнозируем классы для каждой точки в сетке
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Визуализируем границы
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k')
    plt.title(f'Iteration: {iteration}, Loss: {loss:.4f}')
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