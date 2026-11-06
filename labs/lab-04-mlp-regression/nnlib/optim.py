"""Оптимизатор: градиентный спуск, по желанию с моментом."""

import numpy as np


class SGD:
    """Шаг v <- momentum * v + grad, w <- w - lr * v.
    При momentum = 0 это обычный градиентный спуск w <- w - lr * grad."""

    def __init__(self, parameters, lr, momentum=0.0):
        self.parameters = list(parameters)
        self.lr, self.momentum = lr, momentum
        self.velocity = [np.zeros_like(p.data) for p in self.parameters]

    def zero_grad(self):
        for p in self.parameters:
            p.grad.fill(0.0)

    def step(self):
        for p, v in zip(self.parameters, self.velocity):
            v *= self.momentum
            v += p.grad
            p.data -= self.lr * v
