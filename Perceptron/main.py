import numpy as np
from data_generation.generator import get_dataset
from perceptrons.elementaryPeceptron import ElementaryPerceptron
from perceptrons.multilayeredPerceptron import MultilayeredPerceptron as Perceptron
from visualization.visualization import plot_decision_boundary, visualize_loss, plot_data

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def main():
    np.random.seed(21212)

    dataset_choice = 'xor'
    activation_func = 'sigmoid'
    points = 200
    noise = 0
    iterations = 2000
    count_graphics = 1
    display_confussion_matrix = True
    tesing_percent = 0.2
    display_loss = False
    try:
        X, y = get_dataset(dataset_choice, points, noise)
    except ValueError as e:
        print(e)
        return
    # plot_data(X,y)

    perc = Perceptron([2,8,8,1], 1.1, activation_func)
    # perc = ElementaryPerceptron(1.1, activation_func)
    if (display_confussion_matrix): # Show confusion matrix
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=tesing_percent, random_state=42)
        loss_history = perc.fit(X_train, y_train, iterations=iterations, visualization_func=plot_decision_boundary, count_graphics=count_graphics)
        if(display_loss): visualize_loss(loss_history)

        predictions_continuous = np.array([perc.predict(x) for x in X_test])
        predictions_binary = np.where(predictions_continuous > 0.5, 1, 0)

        conf_matrix = confusion_matrix(y_test, predictions_binary)
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
        disp.plot(cmap='Blues')
        disp.ax_.set_title('Confusion Matrix Visualization')
        plt.show()
    else:
        loss_history = perc.fit(X, y, iterations=iterations, visualization_func=plot_decision_boundary, count_graphics=count_graphics)
        if(display_loss): visualize_loss(loss_history)

if __name__ == "__main__":
    main()
