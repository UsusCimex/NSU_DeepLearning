import numpy as np

def func(self, x):
    # Ступенчатая функция активации
    return np.where(x >= 0, 1, 0)