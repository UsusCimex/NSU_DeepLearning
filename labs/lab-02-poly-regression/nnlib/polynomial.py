"""Полиномиальная регрессия методом наименьших квадратов с регуляризацией."""

import numpy as np


def normal_system(t, y, M):
    """Система Aw = b: a_ij = сумма t^(i+j), b_i = сумма t^i * y."""
    A = np.empty((M + 1, M + 1))
    b = np.empty(M + 1)
    for i in range(M + 1):
        b[i] = np.sum(t ** i * y)
        for j in range(M + 1):
            A[i, j] = np.sum(t ** (i + j))
    return A, b


class PolynomialRegression:
    """Многочлен степени degree, коэффициенты w - решение системы (A + lam I_0) w = b,
    I_0 = diag(0, 1, ..., 1). При lam = 0 это метод наименьших квадратов без
    регуляризации."""

    def __init__(self, degree, lam=0.0):
        self.degree, self.lam = degree, lam

    def scale(self, x):
        """Перенос отрезка [x_min, x_max] обучающей выборки в [-1, 1]. Многочлен
        строится от t, а не от x: степени t не больше 1 по модулю, и система решается
        точнее."""
        return 2 * (x - self.x_min) / (self.x_max - self.x_min) - 1

    def fit(self, x, y):
        self.x_min, self.x_max = x.min(), x.max()
        A, b = normal_system(self.scale(x), y, self.degree)
        for j in range(1, self.degree + 1):
            A[j, j] += self.lam
        self.w = np.linalg.solve(A, b)
        return self

    def predict(self, x):
        """Значения многочлена в точках x."""
        t = self.scale(x)
        return sum(self.w[j] * t ** j for j in range(self.degree + 1))
