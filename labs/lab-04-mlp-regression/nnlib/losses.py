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
