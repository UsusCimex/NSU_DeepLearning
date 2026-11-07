"""Генератор случайных чисел библиотеки: начальные веса слоёв и перемешивание батчей."""

import numpy as np

_generator = np.random.default_rng()


def manual_seed(seed):
    """Задаёт начальное значение генератора, чтобы результаты повторялись."""
    global _generator
    _generator = np.random.default_rng(seed)


def generator():
    return _generator
