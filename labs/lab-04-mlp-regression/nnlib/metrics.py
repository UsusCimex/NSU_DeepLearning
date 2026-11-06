"""Метрики качества. pred - ответы модели, target - верные ответы."""

import numpy as np


def mse(pred, target):
    """Средний квадрат отклонения."""
    return float(np.mean((pred - target) ** 2))


def rmse(pred, target):
    """Корень из среднего квадрата отклонения, в единицах target."""
    return float(np.sqrt(mse(pred, target)))


def mae(pred, target):
    """Среднее абсолютное отклонение."""
    return float(np.mean(np.abs(pred - target)))


def r2_score(pred, target):
    """Коэффициент детерминации: 1 - (сумма квадратов ошибок) /
    (сумма квадратов отклонений target от его среднего)."""
    return float(1.0 - np.sum((pred - target) ** 2) / np.sum((target - target.mean()) ** 2))


def accuracy(pred, target):
    """Доля верных ответов, pred и target - номера классов."""
    return float(np.mean(pred == target))


def confusion_matrix(pred, target, n_classes):
    """Матрица ошибок: элемент [i, j] - число объектов класса i с ответом j."""
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(target, pred):
        matrix[t, p] += 1
    return matrix
