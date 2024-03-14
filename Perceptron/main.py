import numpy as np
from data_generation.generator import get_dataset
from perceptrons.elementaryPeceptron import ElementaryPerceptron
from perceptrons.multilayeredPerceptron import MultilayeredPerceptron as Perceptron
from visualization.visualization import plot_decision_boundary, visualize_loss

def main():
    np.random.seed(21212)

    # Выбор датасета
    # dataset_choice = input("Выберите датасет (linear/spiral/xor/cross): ").lower()
    # points = int(input("Количество точек: "))
    # noise = float(input("Разброс точек: "))
    # iterations = int(input("Введите количество итераций в обучении: "))

    dataset_choice = 'xor'
    points = 100
    noise = 0
    iterations = 1000
    try:
        X, y = get_dataset(dataset_choice, points, noise)
    except ValueError as e:
        print(e)
        return

    # Создаем и обучаем нейросеть
    perc = Perceptron([2,8,1], 1, "sigmoid")

    # После обучения нейросети
    loss_history = perc.fit(X, y, iterations=iterations, visualization_func=plot_decision_boundary, count_graphics=2)
    # visualize_loss(loss_history)

if __name__ == "__main__":
    main()
