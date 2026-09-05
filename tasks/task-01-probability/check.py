import math

import numpy as np
from scipy import stats

SEED = 20260905
N = 2_000_000


def task1():
    rng = np.random.default_rng(SEED)
    u = rng.random(N)
    lam = 1.7
    x = -np.log1p(-u) / lam
    ks = stats.kstest(x, 'expon', args=(0, 1 / lam))

    p = np.array([0.5, 0.2, 0.3])
    d = np.searchsorted(np.cumsum(p), u, side='left')
    freq = np.bincount(d, minlength=3) / N

    print('Задание 1')
    print(f'  экспоненциальное: D = {ks.statistic:.5f}, p = {ks.pvalue:.2f}')
    print(f'  дискретное: макс. отклонение частот = {np.abs(freq - p).max():.4f}')


def task2():
    rng = np.random.default_rng(SEED)
    mu, sigma = -3.5, 2.25
    xi = rng.standard_normal(N)
    eta = mu + sigma * xi
    ks = stats.kstest(eta[:100_000], 'norm', args=(mu, sigma))
    back = np.abs((eta - mu) / sigma - xi).max()

    print('Задание 2')
    print(f'  среднее = {eta.mean():.4f} (теория {mu})')
    print(f'  ст. отклонение = {eta.std(ddof=1):.4f} (теория {sigma})')
    print(f'  D = {ks.statistic:.5f}, p = {ks.pvalue:.2f} (100 000 значений)')
    print(f'  макс. отклонение (eta - mu)/sigma от xi = {back:.1e}')


def task3():
    rng = np.random.default_rng(SEED)
    print('Задание 3')
    for n in (1, 2, 12, 30):
        s = rng.random((200_000, n)).sum(axis=1)
        print(f'  n = {n:2d}: M = {s.mean():8.4f} ({n / 2:.4f}), '
              f'D = {s.var(ddof=1):.4f} ({n / 12:.4f})')

    eta = rng.random((500_000, 12)).sum(axis=1) - 6.0
    ks = stats.kstest(eta, 'norm')
    print(f'  нормировка при n = 12: M = {eta.mean():.4f}, '
          f'sigma = {eta.std(ddof=1):.4f}')
    print(f'  сравнение с N(0,1): D = {ks.statistic:.4f}, p = {ks.pvalue:.3f}')


def task4():
    rng = np.random.default_rng(SEED)
    # g(t) = t^3, xi ~ Exp(1), zeta = xi^(1/3)
    xi = rng.exponential(1.0, N)
    zeta = np.cbrt(xi)
    mode_zeta = (2 / 3) ** (1 / 3)

    print('Задание 4')
    print(f'  M xi = {xi.mean():.4f} (теория 1)')
    print(f'  M zeta = {zeta.mean():.4f} (теория {math.gamma(4 / 3):.4f})')
    print(f'  g(M zeta) = {math.gamma(4 / 3) ** 3:.4f}')
    print(f'  мода xi = 0, мода zeta = {mode_zeta:.4f}, '
          f'g(мода zeta) = {mode_zeta ** 3:.4f}')


def task5():
    rng = np.random.default_rng(SEED)
    xi = rng.random(N)
    mono = xi ** 2
    nonmono = (xi - 0.5) ** 2

    print('Задание 5')
    print(f'  M z(xi) = {mono.mean():.4f}, z(M xi) = 0.2500')
    print(f'  медиана z(xi) при z(x) = x^2: {np.median(mono):.4f} (z(1/2) = 0.25)')
    print(f'  медиана z(xi) при z(x) = (x-1/2)^2: {np.median(nonmono):.4f} '
          f'(z(медиана xi) = 0)')


def task6_7_8():
    rng = np.random.default_rng(SEED)
    m = 4
    mu = np.array([1.0, -2.0, 0.5, 3.0])
    sd = np.array([1.0, 2.0, 0.5, 1.5])
    point = np.array([0.3, -1.0, 0.9, 2.0])

    sigma = np.diag(sd ** 2)
    d = point - mu
    joint = float(np.exp(-0.5 * d @ np.linalg.inv(sigma) @ d) /
                  ((2 * np.pi) ** (m / 2) * np.sqrt(np.linalg.det(sigma))))
    product = float(np.prod(np.exp(-0.5 * (d / sd) ** 2) / (sd * np.sqrt(2 * np.pi))))

    u = rng.uniform(-1, 1, N)
    cov = float(np.cov(u, u ** 2)[0, 1])

    a = rng.standard_normal((4, 4))
    s_full = np.cov(a @ rng.standard_normal((4, 50_000)), rowvar=True)

    base = rng.standard_normal((3, 50_000))
    s_deg = np.cov(np.vstack([base, base[0] + 2 * base[1]]), rowvar=True)

    print('Задание 6')
    print(f'  плотность (3) = {joint:.7f}, произведение одномерных = {product:.7f}')
    print(f'  разность = {abs(joint - product):.3e}')
    print('Задание 7')
    print(f'  cov(xi, xi^2) при xi ~ U[-1,1] = {cov:.3e}')
    print('Задание 8')
    print(f'  невырожденный случай: ранг = {np.linalg.matrix_rank(s_full)}, '
          f'min lambda = {np.linalg.eigvalsh(s_full).min():.4f}')
    print(f'  вырожденный случай: ранг = {np.linalg.matrix_rank(s_deg, tol=1e-8)}, '
          f'min lambda = {np.linalg.eigvalsh(s_deg).min():.3e}, '
          f'det = {np.linalg.det(s_deg):.3e}')


def task9():
    rng = np.random.default_rng(SEED)
    prior = np.array([0.2, 0.5, 0.3])
    likelihood = np.array([0.9, 0.1, 0.4])
    evidence = float(prior @ likelihood)
    posterior = prior * likelihood / evidence

    w = rng.choice(3, size=N, p=prior)
    occurred = rng.random(N) < likelihood[w]
    empirical = np.bincount(w[occurred], minlength=3) / occurred.sum()

    print('Задание 9')
    print(f'  P(X) = {evidence:.4f}')
    print(f'  апостериорные по формуле: {np.round(posterior, 4)}')
    print(f'  апостериорные в симуляции: {np.round(empirical, 4)}')
    print(f'  сумма = {posterior.sum():.4f}, '
          f'макс. отклонение = {np.abs(posterior - empirical).max():.1e}')


if __name__ == '__main__':
    for check in (task1, task2, task3, task4, task5, task6_7_8, task9):
        check()
