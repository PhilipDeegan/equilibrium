import numpy as np
import equilibrium as eq
import matplotlib.pyplot as plt

L = 15
Lx = Ly = 15
w1, w2 = 0.2, 1.0


def S(y, y0, l):
    return 0.5 * (1.0 + np.tanh((y - y0) / l))


def bx(x, y):
    x0 = x - 0.5 * Lx
    y1 = y - 0.3 * Ly
    y2 = y - 0.7 * Ly
    w3 = np.exp(-(x0 * x0 + y1 * y1) / (w2 * w2))
    w4 = np.exp(-(x0 * x0 + y2 * y2) / (w2 * w2))
    w5 = 2.0 * w1 / w2
    v1, v2 = -1, 1.0
    return (
        v1
        + (v2 - v1) * (S(y, Ly * 0.3, 0.5) - S(y, Ly * 0.7, 0.5))
        + (-w5 * y1 * w3)
        + (+w5 * y2 * w4)
    )


def by(x, y):
    x0 = x - 0.5 * Lx
    y1 = y - 0.3 * Ly
    y2 = y - 0.7 * Ly
    w3 = np.exp(-(x0 * x0 + y1 * y1) / (w2 * w2))
    w4 = np.exp(-(x0 * x0 + y2 * y2) / (w2 * w2))
    w5 = 2.0 * w1 / w2
    return (w5 * x0 * w3) + (-w5 * x0 * w4)


M = int(L / 2) + 1
d0 = np.linspace(0, L, L)
x, y, z = np.meshgrid(*([d0] * 3))
u = bx(x, y)
w = by(x, y)
v = bx(x, y) * -1
fig = eq.plot.Fig3D(cols=4, rows=3, scale=3)


def sub_plot(data, elev=30, azim=50, roll=0):
    return fig.sub_plot_3d(data, elev, azim, roll)


sub_plot(None, 1, 1, 1).quiver(x, y, z, u, v, w, length=0.5)
# x, y = np.meshgrid(*([d0] * 2))
# fig.sub_plot(None, 4, 4, 1).quiver(x, y, u[:, :, M], w[:, :, M])
# fig.sub_plot(None, 5, 5, 5).quiver(x, y, v[:, :, M], w[:, :, M])
plt.show()
