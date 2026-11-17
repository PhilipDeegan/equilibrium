import numpy as np

import equilibrium as eq
from eq.physical.electrical import electric as el

logger = eq.log.getLogger(__name__)

PI = np.longdouble("3.141592653589793238462643383279502884")
scale_factor = 1e10


class Atom:
    def __init__(self):
        self.p = np.array([0, 0, 0]).astype("double")
        self.r = el.bohr_shell_radius(1, 1) * scale_factor
        self.s = eq.math.geo.sphere.Sphere(self.p, 0.00)
        self.en = 1
        self.ep = np.zeros([self.en, 3]).astype("double")
        self.ev = np.zeros([self.en, 3]).astype("double")

    def col_arrays(self):
        return self.ep[:, 0:1], self.ep[:, 1:2], self.ep[:, 2:3]

    def __repr__(self):
        return str(self.ep) + str(self.ev)


def electrons_energy(n: Atom):
    n.ee = n.r - (n.p - n.ep) * 1e-4
    return n


def electrons_velocity(n: Atom):
    θ = PI / 12
    x, y, z = n.ep[0]
    n.ep[0][0] = np.cos(θ) * x + np.sin(θ) * y
    n.ep[0][1] = -np.sin(θ) * x + np.cos(θ) * y
    return n


def place_electrons(n: Atom, diff=1e-1):
    n.ep[0][0] = n.r
    return n


def tick(n: Atom):
    return electrons_velocity(n)


def aks(n, ax):
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.scatter(*n.col_arrays(), s=4)  # c="b", marker="o")
    return ax


def animate(i, n, ax):
    ax.cla()
    aks(tick(n), ax)
    assert (n.ep > -2.3).all()
    assert (n.ep < 2.3).all()


def main():
    n = place_electrons(Atom())
    fig = eq.plot.Fig3D(cols=4, rows=3, scale=5)
    ax = aks(n, fig.sub_plot_3d(title="electron"))
    eq.plot.Animation(
        fig=fig.fig, func=animate, frames=4, interval=155, fargs=(n, ax)
    ).show()


if __name__ == "__main__":
    main()
