import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from perceptrons.pyTorchPerceptron import MultilayerPerceptron

def main():
    # Генерация данных
    X_train = np.linspace(-5, 5, 15).reshape(-1, 1)
    y_train = np.sin(X_train) + np.random.normal(0, 0.2, X_train.shape)

    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)

    # Создание DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)

    # Инициализация перцептрона с увеличенным числом слоев и нейронов
    layers_sizes = [1, 100, 100, 100, 100, 1]
    learning_rate = 0.001
    activations = [torch.nn.Sigmoid(), torch.nn.Sigmoid(), torch.nn.Sigmoid(), torch.nn.Sigmoid(), None]

    # Создание модели
    model = MultilayerPerceptron(layers_sizes, learning_rate, activations)
    
    # Обучение модели
    model.fit(train_loader, epochs=2000)
    
    # Создаем более плотную сетку для предсказания
    X_test = np.linspace(-5, 5, 1000).reshape(-1, 1)
    y_real = np.sin(X_test)
    X_test_tensor = torch.FloatTensor(X_test)

    # Предсказания модели
    with torch.no_grad():
        predictions = model(X_test_tensor).numpy()
    
    # Визуализация результатов
    plt.scatter(X_train, y_train, label='Исходные данные')
    plt.plot(X_test, predictions, color='red', label='Аппроксимация модели')
    plt.plot(X_test, y_real, label='Исходная функция')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()