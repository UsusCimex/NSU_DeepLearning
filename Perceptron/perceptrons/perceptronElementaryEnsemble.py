import numpy as np
from perceptrons.elementaryPeceptron import ElementaryPerceptron

class PerceptronElementaryEnsemble:
    def __init__(self, perceptrons_activations, learning_rates):
        if (len(perceptrons_activations) != len(learning_rates)):
            raise ValueError("Invalid argument")
        self.perceptrons = [ElementaryPerceptron(learning_rate, activation) for activation, learning_rate in zip(perceptrons_activations, learning_rates)]
    
    def predict(self, X):
        predictions = [p.predict(X) for p in self.perceptrons]
        # print(f"Predictions by all perceptrons: {predictions}")
        majority_vote = np.mean(predictions, axis=0) > 0.5
        final_predictions = majority_vote.astype(int)
        # print(f"Majority vote result: {final_predictions}")
        return final_predictions
    
    
    def fit(self, X, y, iterations=10000, visualization_graph_func=None, count_graphics=1, visualization_loss_func=None):
        n_samples = X.shape[0]
        n_perceptrons = len(self.perceptrons)
        indices = np.random.permutation(n_samples)
        split_indices = np.array_split(indices, n_perceptrons)
        for idx, perceptron in enumerate(self.perceptrons):
            X_part = X[split_indices[idx]]
            y_part = y[split_indices[idx]]
            perceptron.fit(X_part, y_part, iterations=iterations,
                           visualization_graph_func=visualization_graph_func,
                           count_graphics=count_graphics,
                           visualization_loss_func=visualization_loss_func)

    def print_lines(self, X, y, visualization_func):
        lines = []
        for perceptron in self.perceptrons:
            weights = perceptron.weights
            bias = perceptron.bias
            if weights[1] != 0:
                slope = -weights[0] / weights[1]
                intercept = -bias / weights[1]
                lines.append((slope, intercept))
            else:
                print("Warning: weights[1] is zero, line cannot be drawn.")
        visualization_func(X, y, lines)


