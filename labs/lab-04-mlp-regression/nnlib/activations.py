"""Функции активации, применяются к каждому элементу."""

import numpy as np

from .module import Module


class ReLU(Module):
    """ReLU(t) = max(0, t), производная 1 при t > 0 и 0 иначе."""

    def forward(self, x):
        self.positive = x > 0
        return np.where(self.positive, x, 0.0)

    def backward(self, grad):
        return grad * self.positive


class Sigmoid(Module):
    """sigma(t) = 1 / (1 + e^(-t)), производная sigma(t) (1 - sigma(t))."""

    def forward(self, x):
        self.y = 1.0 / (1.0 + np.exp(-x))
        return self.y

    def backward(self, grad):
        return grad * self.y * (1.0 - self.y)


class Tanh(Module):
    """tanh t, производная 1 - tanh^2 t."""

    def forward(self, x):
        self.y = np.tanh(x)
        return self.y

    def backward(self, grad):
        return grad * (1.0 - self.y ** 2)
