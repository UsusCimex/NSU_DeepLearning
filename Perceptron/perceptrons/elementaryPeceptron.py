import numpy as np

import perceptrons.activation_func.relu as relu
import perceptrons.activation_func.sigmoid as sigmoid
import perceptrons.activation_func.step as step
import perceptrons.activation_func.tanh as tanh

class ElementaryPerceptron:
    def __init__(self, learning_rate=0.1, activation='step'):
        self.weights = np.random.randn(2)
        self.bias = 0
        self.learning_rate = learning_rate
        if activation == "sigmoid":
            self.activation_func = sigmoid.func
            self.activation_func_prime = sigmoid.prime
        elif activation == "step":
            self.activation_func = step.func
        else:
            raise Exception("Wrong activation function name!")

    def predict(self, X):
        summation = np.dot(X, self.weights) + self.bias
        return self.activation_func(summation)

    def fit(self, X, y, iterations=100, visualization_graph_func=None, count_graphics=1, visualization_loss_func=None):
        for i in range(1, iterations+1):
            for xx, yy in zip(X, y):
                prediction = self.predict(xx)
                error = yy - prediction
                error = yy - self.activation_func_prime(xx) if self.activation_func_prime is not None else None
                self.weights += self.learning_rate * error * xx
                self.bias += self.learning_rate * error

            if i % (iterations // count_graphics) == 0:
                print(f"Iteration {i}")
                if visualization_graph_func is not None:
                    visualization_graph_func(self, X, y, i)