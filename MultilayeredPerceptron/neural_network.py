import numpy as np

class NeuralNetwork:
    def __init__(self, layers_sizes):
        # Инициализация параметров сети
        self.num_layers = len(layers_sizes)
        self.biases = [np.zeros((1, y)) for y in layers_sizes[1:]]
        self.weights = [np.random.randn(x, y) * 0.01 for x, y in zip(layers_sizes[:-1], layers_sizes[1:])]

    def sigmoid(self, z):
        # Сигмоидальная функция активации
        return 1 / (1 + np.exp(-z))
    
    def sigmoid_derivative(self, z):
        # Производная сигмоидальной функции активации
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def relu(self, z):
        # ReLU функция активации
        return np.maximum(0, z)

    def relu_derivative(self, z):
        # Производная ReLU функции активации
        return (z > 0).astype(float)
    
    def loss(self, y_true, y_pred):
        # Функция потерь (cross-entropy)
        epsilon = 1e-7  # Малая константа для стабилизации логарифма
        return -np.mean(y_true * np.log(y_pred + epsilon) + (1 - y_true) * np.log(1 - y_pred + epsilon))
    
    def forward(self, X, change_value=True):
        # Прямой проход
        activation = X
        activations = [X]  # список для хранения всех активаций
        zs = []  # список для хранения всех векторов z

        for b, w in zip(self.biases, self.weights):
            z = np.dot(activation, w) + b
            zs.append(z)
            activation = self.relu(z) if b is not self.biases[-1] else self.sigmoid(z)
            activations.append(activation)

        if change_value:
            self.activations, self.zs = activations, zs
        return activations[-1]

    def backprop(self, X, y, learning_rate):
        # Обратный проход
        m = y.shape[0]
        self.forward(X)  # Прямой проход для получения активаций и z-значений
        deltas = [None] * len(self.weights)  # разница между предсказанным и реальным значением

        # Последний слой
        delta = self.activations[-1] - y
        deltas[-1] = delta

        # Распространение ошибки назад
        for l in range(2, self.num_layers):
            z = self.zs[-l]
            sp = self.sigmoid_derivative(z) if l == 2 else self.relu_derivative(z)
            delta = np.dot(delta, self.weights[-l+1].T) * sp
            deltas[-l] = delta

        # Обновление весов
        self.weights = [w - (1/m) * np.dot(act.T, delta) * learning_rate for w, act, delta in zip(self.weights, self.activations[:-1], deltas)]
        self.biases = [b - np.mean(delta, axis=0, keepdims=True) * learning_rate for b, delta in zip(self.biases, deltas)]

    def train(self, X, y, iterations=10000, learning_rate=1.2, visualization_func=None):
        # Обучение модели
        loss_history = []
        for i in range(iterations):
            y_pred = self.forward(X)
            loss = self.loss(y, y_pred)
            loss_history.append(loss)

            if i % (iterations // 5) == 0:
                print(f"Iteration {i}, Loss: {loss}")
                if visualization_func is not None:
                    visualization_func(self, X, y, i, loss)

            self.backprop(X, y, learning_rate)
        
        print(f"Iteration {iterations}, Loss: {loss_history[-1]}")
        if visualization_func is not None:
            visualization_func(self, X, y, iterations, loss_history[-1])
        return loss_history
    
    def predict(self, X):
        # Предсказание
        y_pred = self.forward(X, change_value=False)
        return (y_pred > 0.5).astype(int)
