import numpy as np
from data_generation.generator import get_dataset
from perceptrons.elementaryPeceptron import ElementaryPerceptron
from perceptrons.perceptronElementaryEnsemble import PerceptronElementaryEnsemble
from perceptrons.multilayeredPerceptron import MultilayeredPerceptron
from visualization.visualization import visualize_graph, visualize_loss, plot_data, plot_lines

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def to_polar(x, y):
    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return r, phi

def main():
    np.random.seed(21212)

    dataset_choice = 'linear_circle'
    activation_func = 'sigmoid'
    points = 200
    noise = 0.2
    iterations = 500
    count_graphics = 5
    display_confusion_matrix = False
    testing_percent = 0.2
    display_loss = False
    display_graph = True
    polar_system = False
    try:
        X, y = get_dataset(dataset_choice, points, noise)
    except ValueError as e:
        print(e)
        return
    plot_data(X,y)

    if (polar_system):
        polar_X = np.array([to_polar(x[0], x[1]) for x in X])
        X = polar_X
        plot_data(X,y)

    # perceptron = MultilayeredPerceptron([2,8,8,8,1], 0.3, activation_func)
    perceptron = ElementaryPerceptron(0.3, activation=activation_func, method='gradient')
    # perceptron = PerceptronElementaryEnsemble(['sigmoid', 'step', 'sigmoid'], [0.1, 0.05, 0.03])
    if (display_confusion_matrix): # Show confusion matrix
        perceptron.fit(X, y, iterations=iterations, 
            visualization_graph_func=(visualize_graph if display_graph else None), 
            visualization_loss_func=(visualize_loss if display_loss else None), 
            count_graphics=count_graphics)
        
        if perceptron.__class__ is PerceptronElementaryEnsemble:
            perceptron.print_lines(X, y, plot_lines)

        predictions_continuous = np.array([perceptron.predict(x) for x in X])
        predictions_binary = np.where(predictions_continuous > 0.5, 1, 0)

        conf_matrix = confusion_matrix(y, predictions_binary)
        display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
        display.plot(cmap='Blues')
        display.ax_.set_title('Confusion Matrix Visualization')
        plt.show()
    else:
        perceptron.fit(X, y, iterations=iterations, 
            visualization_graph_func=(visualize_graph if display_graph else None), 
            count_graphics=count_graphics,
            visualization_loss_func=(visualize_loss if display_loss else None))

if __name__ == "__main__":
    main()
