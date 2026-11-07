"""Метрики качества. pred - ответы модели, target - верные ответы."""

import numpy as np


def accuracy(pred, target):
    """Доля верных ответов, pred и target - номера классов."""
    return float(np.mean(pred == target))


def confusion_matrix(pred, target, n_classes):
    """Матрица ошибок: элемент [i, j] - число объектов класса i с ответом j."""
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(target, pred):
        matrix[t, p] += 1
    return matrix
