import numpy as np

def func(self, x):
    # ReLU функция активации
    return np.maximum(0, x)

def prime(self, x):
    # Производная ReLU функции активации
    return (x > 0).astype(float)
