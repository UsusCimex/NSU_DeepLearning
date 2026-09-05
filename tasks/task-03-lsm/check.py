import numpy as np

SEED = 20260905


def task1():
    rng = np.random.default_rng(SEED)
    N = 12
    x = rng.uniform(-2, 2, N)
    y = 3 * x - 1 + rng.normal(0, 0.3, N)
    G = np.array([[np.sum(x ** 2), np.sum(x)], [np.sum(x), N]])

    print('Задание 1')
    print(f'  det G = {np.linalg.det(G):.4f}, '
          f'N*sum(x - xmean)^2 = {N * np.sum((x - x.mean()) ** 2):.4f}')
    a, b = np.linalg.solve(G, [np.sum(x * y), np.sum(y)])
    print(f'  A = {a:.4f}, B = {b:.4f} (истинные 3 и -1)')

    x0 = 1.7
    xc = np.full(N, x0)
    yc = 3 * x0 - 1 + rng.normal(0, 0.3, N)
    Gc = np.array([[np.sum(xc ** 2), np.sum(xc)], [np.sum(xc), N]])
    err = lambda A, B: 0.5 * np.sum((yc - A * xc - B) ** 2)
    line = [(A, yc.mean() - A * x0) for A in (0.0, 3.0, -7.5)]
    print(f'  вырожденный случай (все x = {x0}): det = {np.linalg.det(Gc):.3e}, '
          f'ранг = {np.linalg.matrix_rank(Gc)}')
    print('  E(A, B) на прямой A*x0 + B = ymean: '
          + ', '.join(f'E({A}, {B:.3f}) = {err(A, B):.6f}' for A, B in line))


def task2():
    rng = np.random.default_rng(SEED)
    N = 50
    x = rng.uniform(0, 1, N)
    y = np.cos(x) + rng.normal(0, 0.05, N)

    print('Задание 2')
    print('  число обусловленности A при N = 50 различных узлах')
    for M in (0, 1, 3, 5, 10, 15):
        V = np.vander(x, M + 1, increasing=True)
        A = V.T @ V
        print(f'    M = {M:2d}: cond = {np.linalg.cond(A):.3e}, '
              f'ранг = {np.linalg.matrix_rank(A)} из {M + 1}')

    print('  мало различных узлов: r = 4, каждый повторён 5 раз')
    xr = np.repeat(np.array([0.1, 0.3, 0.6, 0.9]), 5)
    yr = np.cos(xr) + rng.normal(0, 0.05, xr.size)
    for M in (3, 5, 8):
        V = np.vander(xr, M + 1, increasing=True)
        A, b = V.T @ V, V.T @ yr
        rank = np.linalg.matrix_rank(A)
        w = np.linalg.lstsq(A, b, rcond=None)[0]
        print(f'    M = {M}: ранг A = {rank} из {M + 1}, '
              f'размерность ядра = {M + 1 - rank}, '
              f'невязка ||Aw - b|| = {np.linalg.norm(A @ w - b):.2e}')


if __name__ == '__main__':
    task1()
    task2()
