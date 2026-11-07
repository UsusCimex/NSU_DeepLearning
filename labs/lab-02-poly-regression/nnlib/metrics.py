"""Метрики качества. pred - ответы модели, target - верные ответы."""

import numpy as np


def rmse(pred, target):
    """Корень из среднего квадрата отклонения, в единицах target."""
    return float(np.sqrt(np.mean((pred - target) ** 2)))
