import numpy as np
import matplotlib.pyplot as plt
from numpy import random

# Task 1
random.seed(21212)
eps0 = 0.5
N = 5

# Function A
def func_a(x):
   return x * np.sin(2 * np.pi * x)

# Function B
a = random.randint(-3, 3)
b = random.randint(-3, 3)
c = random.randint(-3, 3)
d = random.randint(-3, 3)
print(f"Selected: a={a} b={b} c={c} d={d}")
def func_b(x):
    return a * x ** 3 + b * x ** 2 + c * x + d

# Distribution function
def y(f, x, distr):
    if distr == 'uniform':
        return f(x) + random.uniform(-eps0, eps0)
    elif distr == 'normal':
        return f(x) + random.normal(loc=0, scale=eps0)
    else:
        return Error

defineX = np.linspace(-1, 1, 100)
defineY = func_a(defineX)

xi = np.array([random.uniform(-1, 1) for _ in range(N)])
yi = np.array([y(func_a, x, 'uniform') for x in xi])

# Task 2

def compute_coefficients(x, y, degree):
    N = len(x)
    M = degree + 1

    A = np.zeros((M, M))
    for i in range(M):
        for j in range(M):
            A[i, j] = 0
            for k in range(N):
                A[i, j] += x[k] ** (i + j)

    b = np.zeros(M)
    for i in range(M):
        b[i] = 0
        for k in range(N):
            b[i] += y[k] * x[k] ** i

    coefficients = np.linalg.solve(A, b)
    return coefficients

coefficients = compute_coefficients(xi, yi, degree=10)
def fitted_function(x):
    result = 0
    for i, coef in enumerate(coefficients):
        result += coef * x ** i
    return result

plt.figure(figsize=(10, 6))
plt.scatter(xi, yi, color='red', label='Выборка', marker='o')
plt.plot(defineX, fitted_function(defineX), label='Функция, полученная регрессией', color='green')
plt.plot(defineX, defineY, label='Исходная функция', color='red')
plt.ylim(-5, 5)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Аппроксимация функции методом наименьших квадратов')
plt.legend()
plt.grid(True)
# plt.savefig('10b1.png')
plt.show()