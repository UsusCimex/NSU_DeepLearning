import numpy as np 

def func(x):
    # Сигмоидальная функция активации
    return 1 / (1 + np.exp(-x))

def prime(x):
    # Производная сигмоидальной функции активации
    return func(x) * (1 - func(x))