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
    
    def fit(self, X, y, iterations=100, visualization_func=None, count_graphics=1):
        for perceptron in self.perceptrons:
            perceptron.fit(X, y, iterations=iterations, visualization_func=visualization_func, count_graphics=count_graphics)
