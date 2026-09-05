import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEED = 20260905


def knn_regression(x_train, y_train, x_query, k):
    d = np.abs(x_query[:, None] - x_train[None, :])
    idx = np.argsort(d, axis=1)[:, :k]
    return y_train[idx].mean(axis=1)


def task1():
    rng = np.random.default_rng(SEED)
    n = 45
    x = np.sort(rng.uniform(0, 1, n))
    y = x + rng.normal(0, 0.06, n)
    grid = np.linspace(0, 1, 2000)

    print('Задание 1')
    for k in (1, 2, 3, 5):
        fit = knn_regression(x, y, x, k)
        err = np.abs(fit - y).max()
        curve = knn_regression(x, y, grid, k)
        jumps = int(np.sum(np.abs(np.diff(curve)) > 1e-12))
        print(f'  k = {k}: макс. отклонение кривой от точек = {err:.4f}, '
              f'ступеней = {jumps + 1}')

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    for ax, k in zip(axes, (1, 5)):
        ax.plot(grid, knn_regression(x, y, grid, k), color='tab:blue',
                label=f'kNN, k = {k}')
        ax.plot(grid, grid, color='black', label='истинная зависимость')
        ax.scatter(x, y, color='darkred', s=18, zorder=3, label='выборка')
        ax.set_title(f'k = {k}')
        ax.set_xlabel('x')
        ax.grid(alpha=0.3)
    axes[0].set_ylabel('y')
    axes[0].legend(loc='upper left', fontsize=8)
    fig.tight_layout()
    fig.savefig('knn-regression.png', dpi=150)


def task2():
    print('Задание 2')
    n, M = 10, 100
    for m in (12, 20, 40):
        N = M * n ** m
        print(f'  m = {m}: ячеек {n}^{m} = 1e{m}, N = {N:.1e}')
    bytes20 = M * n ** 20 * 20 * 8
    print(f'  память для m = 20 при float64: {bytes20:.1e} байт')


def task3a():
    rng = np.random.default_rng(SEED)
    print('Задание 3а, доля точек в приграничном слое толщины eps')
    for m in (5, 20):
        x = rng.random((200_000, m))
        for eps in (0.01, 0.05):
            near = np.mean(np.any((x < eps) | (x > 1 - eps), axis=1))
            print(f'  m = {m:2d}, eps = {eps:.2f}: опыт {near:.4f}, '
                  f'теория {1 - (1 - 2 * eps) ** m:.4f}')


def task3b():
    rng = np.random.default_rng(SEED)
    print('Задание 3б, длина ребра подкуба, содержащего долю p точек')
    for m in (5, 20):
        x = rng.random((200_000, m))
        for p in (0.01, 0.1):
            l = p ** (1 / m)
            inside = np.mean(np.all(x <= l, axis=1))
            print(f'  l_{m}({p}) = {l:.4f}, доля точек в подкубе: '
                  f'опыт {inside:.4f}, теория {p}')


def task3v():
    rng = np.random.default_rng(SEED)
    print('Задание 3в, евклидовы расстояния между точками')
    for m in (1, 5, 20, 100):
        a = rng.random((2000, m))
        b = rng.random((2000, m))
        d = np.linalg.norm(a - b, axis=1)
        print(f'  m = {m:3d}: среднее {d.mean():.4f}, '
              f'sqrt(M d^2) = {np.sqrt(m / 6):.4f}, '
              f'sigma/среднее {d.std() / d.mean():.4f}, '
              f'(max - min)/min = {(d.max() - d.min()) / d.min():.4f}')

    print('  доля точек в шаре, вписанном в единичный куб')
    from math import lgamma, log, pi, exp
    for m in (2, 5, 20):
        log_v = (m / 2) * log(pi) - lgamma(m / 2 + 1) - m * log(2)
        x = rng.random((200_000, m))
        inside = np.mean(np.linalg.norm(x - 0.5, axis=1) <= 0.5)
        print(f'    m = {m:2d}: теория {exp(log_v):.3e}, опыт {inside:.5f}')


def task3g():
    rng = np.random.default_rng(SEED)
    print('Задание 3г, расстояния Чебышёва')
    for m in (5, 20, 100):
        a = rng.random((200_000, m))
        b = rng.random((200_000, m))
        d = np.abs(a - b).max(axis=1)
        med = 1 - np.sqrt(1 - 2 ** (-1 / m))
        print(f'  m = {m:3d}: среднее {d.mean():.4f}, медиана {np.median(d):.4f} '
              f'(теория {med:.4f})')


if __name__ == '__main__':
    for check in (task1, task2, task3a, task3b, task3v, task3g):
        check()
