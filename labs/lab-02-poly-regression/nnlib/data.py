"""Разбиение выборки для кросс-валидации."""

import numpy as np


def kfold(n, folds, rng):
    """Номера точек 0, ..., n - 1 в случайном порядке, разбитые на folds частей почти
    равного размера. rng - генератор случайных чисел numpy."""
    return np.array_split(rng.permutation(n), folds)
