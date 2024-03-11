import numpy as np
from data_generation import get_dataset
from neural_network import NeuralNetwork
from visualization import plot_decision_boundary, visualize_loss

def main():
    np.random.seed(21212)

    # Выбор датасета
    # dataset_choice = input("Выберите датасет (linear/spiral/xor/cross): ").lower()
    # points = int(input("Количество точек: "))
    # noise = float(input("Разброс точек: "))
    # interations = int(input("Введите количество итераций в обучении: "))

    dataset_choice = 'spiral'
    points = 300
    noise = 0
    interations = 20000
    try:
        X, Y = get_dataset(dataset_choice, points, noise)
    except ValueError as e:
        print(e)
        return

    # Преобразуем Y в формат, подходящий для двоичной классификации
    Y = Y.reshape(Y.shape[0], 1)

    # Создаем и обучаем нейросеть
    nn = NeuralNetwork([2,8,8,8,1])

    # После обучения нейросети
    loss_history = nn.train(X, Y, iterations=interations, learning_rate=1.2, visualization_func=plot_decision_boundary)
    visualize_loss(loss_history)

if __name__ == "__main__":
    main()
