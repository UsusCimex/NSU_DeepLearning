"""Оптимизатор: градиентный спуск."""


class SGD:
    """Шаг градиентного спуска w <- w - lr * grad для каждого параметра."""

    def __init__(self, parameters, lr):
        self.parameters = list(parameters)
        self.lr = lr

    def zero_grad(self):
        for p in self.parameters:
            p.grad.fill(0.0)

    def step(self):
        for p in self.parameters:
            p.data -= self.lr * p.grad
