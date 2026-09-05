import numpy as np

import perceptrons.activation_func.relu as relu
import perceptrons.activation_func.sigmoid as sigmoid
import perceptrons.activation_func.step as step
import perceptrons.activation_func.tanh as tanh

class ElementaryPerceptron:
    def __init__(self, learning_rate=0.1, activation='step', method='gradient'):
        self.method = method
        self.weights = np.random.randn(2)
        self.bias = 0
        self.learning_rate = learning_rate
        if activation == "sigmoid":
            self.activation_func = sigmoid.func
            self.activation_func_prime = sigmoid.prime
        elif activation == "step":
            self.activation_func = step.func
            self.activation_func_prime = lambda x : 1
        elif activation == "relu":
            self.activation_func = relu.func
            self.activation_func_prime = relu.prime
        else:
            raise Exception("Wrong activation function name!")

    def predict(self, X):
        summation = np.dot(X, self.weights) + self.bias
        return (self.activation_func(summation) > 0.5).astype(int)

    def fit(self, X, y, iterations=100, visualization_graph_func=None, count_graphics=1, visualization_loss_func=None):
        for i in range(1, iterations+1):
            for xx, yy in zip(X, y):
                summation = np.dot(xx, self.weights) + self.bias
                activation = self.activation_func(summation)
                if self.method == 'gradient':
                    error = yy - activation
                    gradient = error * self.activation_func_prime(summation)
                    self.weights += self.learning_rate * gradient * xx
                    self.bias += self.learning_rate * gradient
                else:
                    prediction = int(activation > 0.5)
                    if prediction != yy:
                        update = self.learning_rate * (yy - prediction)
                        self.weights += update * xx
                        self.bias += update

            if i % (iterations // count_graphics) == 0:
                print(f"Iteration {i}: Weights={self.weights}, Bias={self.bias}")
                if visualization_graph_func:
                    visualization_graph_func(self, X, y, iteration=i)