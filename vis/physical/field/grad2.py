import numpy as np

import equilibrium as eq


def fn(x, y):
    return np.sin(x) - np.cos(y)


def dx(x, y, diff=1e-5):
    return (fn(x + diff, y) - fn(x - diff, y)) / (2 * diff)


def dy(x, y, diff=1e-5):
    return (fn(x, y + diff) - fn(x, y - diff)) / (2 * diff)


def main():
    eq.plot.style.use("dark_background")
    N = 111
    X = np.linspace(0, N, N - 1)
    Y = np.linspace(0, N, N - 1)

    F = fn(X, Y)
    Dx = dx(X, Y)
    Dy = dy(X, Y)
    G = Dx + Dy
    Dx2 = dx(Dx, Y)
    Dy2 = dy(X, Dy)
    L = Dx2 + Dy2
    fig = eq.plot.Fig(3, 3, 4)

    def subplot(*args, **kwargs):
        fig.sub_plot().plot(*args)

    subplot(F)
    subplot(Dx)
    subplot(Dy)
    subplot(G)
    subplot(Dx2)
    subplot(Dy2)
    subplot(L)

    fig.show()


if __name__ == "__main__":
    main()
