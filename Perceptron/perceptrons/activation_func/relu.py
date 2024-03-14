import numpy as np

def func(x):
    # ReLU функция активации
    return np.maximum(0, x)

def prime(x):
    # Производная ReLU функции активации
    return (x > 0).astype(float)
