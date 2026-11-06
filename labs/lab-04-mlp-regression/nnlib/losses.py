"""Функции потерь. Вызов считает потери на батче, backward возвращает их градиент
по выходам сети."""

import numpy as np


class MSELoss:
    """Средний квадрат отклонения: L = (1/n) * сумма (pred - target)^2."""

    def __call__(self, pred, target):
        if pred.shape != target.shape:
            raise ValueError(f'размеры pred {pred.shape} и target {target.shape} различаются')
        self.diff = pred - target
        return float(np.mean(self.diff ** 2))

    def backward(self):
        return 2.0 * self.diff / self.diff.size


class CrossEntropyLoss:
    """Мягкий максимум и перекрёстная энтропия. pred - выходы сети (n, k),
    target - номера классов: L = -(1/n) * сумма log p[n, target[n]]."""

    def __call__(self, pred, target):
        z = pred - pred.max(axis=1, keepdims=True)  # сдвиг не меняет p, но exp не переполняется
        log_p = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
        self.p, self.target = np.exp(log_p), target
        rows = np.arange(len(target))
        return float(-np.mean(log_p[rows, target]))  # из строки n берётся столбец target[n]

    def backward(self):
        grad = self.p.copy()
        grad[np.arange(len(self.target)), self.target] -= 1.0  # p минус one-hot вектор класса
        return grad / len(self.target)
