"""Функции активации, применяются к каждому элементу."""

import numpy as np

from .module import Module


class Tanh(Module):
    """tanh t, производная 1 - tanh^2 t."""

    def forward(self, x):
        self.y = np.tanh(x)
        return self.y

    def backward(self, grad):
        return grad * (1.0 - self.y ** 2)
