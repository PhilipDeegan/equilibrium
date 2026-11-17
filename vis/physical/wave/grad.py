import numpy as np

import equilibrium as eq


def fn(x, y, z):
    PI = np.longdouble("3.141592653589793238462643383279502884")
    return np.cos(x * PI / 10) + np.cos(y * PI / 20) + np.cos(z * PI / 30)


def dx(x, y, z, diff=1e-5):
    return (fn(x + diff, y, z) - fn(x - diff, y, z)) / (2 * diff)


def dy(x, y, z, diff=1e-5):
    return (fn(x, y + diff, z) - fn(x, y - diff, z)) / (2 * diff)


def dz(x, y, z, diff=1e-5):
    return (fn(x, y, z + diff) - fn(x, y, z - diff)) / (2 * diff)


def main():
    eq.plot.style.use("dark_background")

    N = 100
    X = np.linspace(0, N, N - 1)
    Y = np.linspace(0, N, N - 1)
    Z = np.linspace(0, N, N - 1)

    F = fn(X, Y, Z)
    Dx = dx(X, Y, Z)
    Dy = dy(X, Y, Z)
    Dz = dz(X, Y, Z)
    G = Dx + Dy + Dz
    Dx2 = dx(Dx, Y, Z)
    Dy2 = dy(X, Dy, Z)
    Dz2 = dz(X, Y, Dz)
    L = Dx2 + Dy2 + Dz2
    fig = eq.plot.Fig3D(3, 3, 4)

    def subplot(*args, **kwargs):
        fig.sub_plot().plot(*args)

    subplot(F)
    subplot(G)
    subplot(L)
    subplot(Dx)
    subplot(Dy)
    subplot(Dz)
    subplot(Dx2)
    subplot(Dy2)
    subplot(Dz2)

    fig.show()


if __name__ == "__main__":
    main()
