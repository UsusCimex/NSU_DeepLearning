"""Базовые классы: обучаемый параметр и слой."""

import numpy as np


class Parameter:
    """Обучаемый массив и градиент функции потерь по нему."""

    def __init__(self, data):
        self.data = np.asarray(data, dtype=float)
        self.grad = np.zeros_like(self.data)


class Module:
    """Слой или сеть. forward по батчу входов (строка - один объект) считает выходы
    и запоминает то, что нужно для backward. backward получает градиент функции
    потерь по выходам, добавляет градиенты по параметрам в их grad и возвращает
    градиент по входам."""

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError

    def parameters(self):
        return []

    def zero_grad(self):
        for p in self.parameters():
            p.grad.fill(0.0)

    def __repr__(self):
        return f'{type(self).__name__}()'
