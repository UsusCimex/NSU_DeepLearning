import numpy as np

def func(x):
    # Функция активации тангенс
    return np.tanh(x)

def prime(x):
    # Производная тангенса
    return 1 - np.tanh(x)**2