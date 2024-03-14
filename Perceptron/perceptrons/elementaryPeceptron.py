import numpy as np

import perceptrons.activation_func.relu as relu
import perceptrons.activation_func.sigmoid as sigmoid
import perceptrons.activation_func.step as step
import perceptrons.activation_func.tanh as tanh

class ElementaryPerceptron:
    def __init__(self, input_size, learning_rate=0.1, activation='step'):
        self.weights = np.random.randn(input_size + 1)  # +1 for the bias
        self.learning_rate = learning_rate
        if activation == 'step':
            self.activation_func = step_function
        elif activation == 'sigmoid':
            self.activation_func = self.sigmoid
        else:
            raise ValueError("Activation function not supported.")

    def predict(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        return self.activation_func(summation)

    def train(self, training_inputs, labels):
        for _ in range(100):  # number of epochs could be parameterized
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                self.weights[1:] += self.learning_rate * (label - prediction) * inputs
                self.weights[0] += self.learning_rate * (label - prediction)