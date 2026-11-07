"""Линейный слой."""

import numpy as np

from .module import Module, Parameter
from .random import generator


class Linear(Module):
    """Линейный слой y = x W^T + b. W[j, i] - вес связи входа i с выходом j."""

    def __init__(self, in_features, out_features):
        self.in_features, self.out_features = in_features, out_features
        # начальные веса случайны, дисперсия 1 / (число входов)
        std = np.sqrt(1.0 / in_features)
        self.weight = Parameter(generator().normal(0.0, std, (out_features, in_features)))
        self.bias = Parameter(np.zeros(out_features))

    def forward(self, x):
        self.x = x
        return x @ self.weight.data.T + self.bias.data  # bias прибавляется к каждой строке

    def backward(self, grad):
        # grad[n, j] - производная потерь по выходу j на объекте n; произведение
        # матриц складывает grad[n, j] * x[n, i] по всем объектам батча
        self.weight.grad += grad.T @ self.x
        self.bias.grad += grad.sum(axis=0)
        return grad @ self.weight.data  # по входу i: сумма по j grad[n, j] * W[j, i]

    def parameters(self):
        return [self.weight, self.bias]

    def __repr__(self):
        return f'Linear(in_features={self.in_features}, out_features={self.out_features})'
