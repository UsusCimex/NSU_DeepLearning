import numpy as np
from perceptrons.elementaryPeceptron import ElementaryPerceptron

class PerceptronElementaryEnsemble:
    def __init__(self, perceptrons_activations, learning_rates):
        if (len(perceptrons_activations) != len(learning_rates)):
            raise ValueError("Invalid argument")
        self.perceptrons = [ElementaryPerceptron(learning_rate, activation) for activation, learning_rate in zip(perceptrons_activations, learning_rates)]
    
    def predict(self, X):
        # Получаем предсказания от всех перцептронов и усредняем их
        predictions = np.mean([p.predict(X) for p in self.perceptrons], axis=0)
        return predictions
    
    def fit(self, X, y, iterations=10000, visualization_graph_func=None, count_graphics=1, visualization_loss_func=None):
        for perceptron in self.perceptrons:
            perceptron.fit(X, y, iterations=iterations, 
                visualization_graph_func=visualization_graph_func, 
                count_graphics=count_graphics, 
                visualization_loss_func=visualization_loss_func)

    def print_lines(self, X, y, visualization_func):
        lines = []
        for perceptron in self.perceptrons:
            weights = perceptron.weights
            bias = perceptron.bias
            line = (-weights[0] / weights[1], -bias / weights[1])  # Equation of line: w1*x + w2*y + b = 0
            lines.append(line)
        visualization_func(X, y, lines)

