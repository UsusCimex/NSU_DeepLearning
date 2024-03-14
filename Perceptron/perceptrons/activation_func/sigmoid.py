import numpy as np 

def func(self, x):
    # Сигмоидальная функция активации
    return 1 / (1 + np.exp(-x))

def prime(self, x):
    # Производная сигмоидальной функции активации
    return self.sigmoid(x) * (1 - self.sigmoid(x))