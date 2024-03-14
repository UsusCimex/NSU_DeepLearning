import numpy as np

import perceptrons.activation_func.relu as relu
import perceptrons.activation_func.sigmoid as sigmoid
import perceptrons.activation_func.step as step
import perceptrons.activation_func.tanh as tanh

class MultilayeredPerceptron:
    def __init__(self, layers_sizes, learning_rate, activation):
        # Инициализация параметров сети
        if activation == "sigmoid":
            self.activation_func = sigmoid.func
            self.activation_func_prime = sigmoid.prime
        elif activation == "relu":
            self.activation_func = relu.func
            self.activation_func_prime = relu.prime
        elif activation == "step":
            self.activation_func = step.func
            # Замечание: у ступенчатой функции нет производной
        elif activation == "tanh":
            self.activation_func = tanh.func
            self.activation_func_prime = tanh.prime
        else:
            raise Exception("Wrong activation function name!")

        self.learning_rate = learning_rate
        self.num_layers = len(layers_sizes)
        self.biases = [0 for _ in layers_sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(layers_sizes[:-1], layers_sizes[1:])]

    def forward(self, X):
        # Прямое распространение
        activations = [X.T]
        zs = []
        for i, (b, w) in enumerate(zip(self.biases, self.weights)):
            # print(f"Layer {i}:")
            # print(f"Activations shape: {activations[-1].shape}, Weights shape: {w.shape}, Biases shape: {b.shape}")
            z = np.dot(w, activations[-1]) + b
            a = self.activation_func(z)
            zs.append(z)
            activations.append(a)
        return activations, zs

    def backpropagation(self, X, y):
        nabla_b = [0 for _ in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # Прямое распространение
        activations, zs = self.forward(X)
        
        # Ошибка на выходном слое
        delta = (activations[-1] - y) * self.activation_func_prime(zs[-1])
        nabla_b[-1] = np.sum(delta)
        nabla_w[-1] = np.dot(delta, activations[-2].T)

        # Обратное распространение ошибки
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = self.activation_func_prime(z)
            delta = np.dot(self.weights[-l+1].T, delta) * sp
            nabla_b[-l] = np.sum(delta) 
            nabla_w[-l] = np.dot(delta, activations[-l-1].T)
        
        # Обновление весов и смещений
        self.weights = [w-(self.learning_rate/len(X))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (self.learning_rate / len(X)) * nb for b, nb in zip(self.biases, nabla_b)]

    def fit(self, X, y, iterations=10000, visualization_func=None, count_graphics=4):
        # Обучение модели
        # print("X:" + str(X.shape))
        # print("y:" + str(y.shape))
        loss_history = []
        for i in range(iterations + 1):
            # Прямое распространение
            activations, zs = self.forward(X)
            # Вычисление потерь
            loss = self.loss(y, activations[-1])
            loss_history.append(loss)

            if i != 0 and i % (iterations // count_graphics) == 0:
                print(f"Iteration {i}, Loss: {loss}")
                if visualization_func is not None:
                    visualization_func(self, X, y, i, loss)

            # Обратное распространение для обновления весов и смещений
            self.backpropagation(X, y)
        return loss_history

    def loss(self, y_true, y_pred):
        # Функция потерь (cross-entropy)
        epsilon = 1e-7  # Малая константа для стабилизации логарифма
        return -np.mean(y_true * np.log(y_pred + epsilon) + (1 - y_true) * np.log(1 - y_pred + epsilon))

    def predict(self, X):
        a, _ = self.forward(X)
        return a[-1]
        
