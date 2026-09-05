import numpy as np

def func(x):
    # Ступенчатая функция активации
    return np.where(x >= 0, 1, 0)