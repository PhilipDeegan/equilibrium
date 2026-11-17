import sys
import logging
import numpy as np
import random as rand
import equilibrium as eq
from itertools import combinations
from equilibrium.log import getLogger


logger = getLogger(__name__)

PI = np.longdouble("3.141592653589793238462643383279502884")


class Nuke:
    def __init__(self):
        self.p = np.array([0, 0, 0]).astype("double")
        self.r = 0.1
        self.s = eq.math.geo.sphere.Sphere(self.p, 0.00)
        self.en = 100
        self.ep = np.ones([self.en, 3]).astype("double")
        self.ev = np.zeros([self.en, 3]).astype("double")
        self.ee = np.zeros([self.en, 3]).astype("double")

        self.pp = np.zeros([1, 3]).astype("double")
        self.pv = np.zeros([1, 3]).astype("double")
        self.pp[:] = 1

        self.es = 0.005

    def col_arrays(self):
        return n.ep[:, 0:1], n.ep[:, 1:2], n.ep[:, 2:3]

    def overlaps(self, i, j):
        return (self.ep[i] - self.ep[j] < self.es + self.es).all()


def cloud_points_energy(n: Nuke):
    n.ee = n.r - (n.p - n.ep) * 1e-4
    return n


def cloud_points_velocity(n: Nuke):
    θ = PI / 32
    x = np.copy(n.ep[:, 0])
    y = np.copy(n.ep[:, 1])
    z = np.copy(n.ep[:, 2])
    n.ep[:, 0] = x * np.cos(θ) - y * np.sin(θ)  # - z * np.tan(θ)
    n.ep[:, 1] = x * np.sin(θ) + y * np.cos(θ)  # + z * np.tan(θ)
    n.ep[:, 2] = z * np.cos(θ) - y * np.sin(θ)  # - z * np.sin(θ)
    return n


def place_electron_cloud_points(n: Nuke, diff=1e-1):
    s = sign = np.random.uniform(low=-1, high=1, size=n.ep.shape)
    n.ep = np.random.uniform(low=-n.r - diff, high=n.r + diff, size=n.ep.shape)
    n.ep *= s / abs(s)
    return cloud_points_energy(cloud_points_velocity(n))


def tick_electron_cloud_point_collisions(n: Nuke):
    def change_velocities(i, j, damp=1e-4):
        p0, p1 = n.ep[i], n.ep[j]
        v0, v1 = n.ev[i], n.ev[j]
        m1, m2 = n.es**2, n.es**2
        M = m1 + m2
        d = np.linalg.norm(p0 - p1) ** 2
        u1 = v0 - 2 * m2 / M * np.dot(v0 - v1, p0 - p1) / d * (p0 - p1) * damp
        u2 = v1 - 2 * m1 / M * np.dot(v1 - v0, p1 - p0) / d * (p1 - p0) * damp
        n.ev[i] = u1
        n.ev[j] = u2

    for i, j in combinations(range(n.en), 2):
        if n.overlaps(i, j):
            logger.info("collision")
            change_velocities(i, j)
    return n


def tick_electron_cloud_points(n: Nuke):
    n = tick_electron_cloud_point_collisions(n)
    n.ep += n.ev
    return cloud_points_velocity(n)


def aks(n, ax):
    ax.set_xlim(-0.4, 0.4)
    ax.set_ylim(-0.4, 0.4)
    ax.set_zlim(-0.4, 0.4)
    ax.scatter(*n.col_arrays(), s=4)  # c="b", marker="o")
    return ax


def animate(i, n, ax):
    ax.cla()
    aks(tick_electron_cloud_points(n), ax)
    assert (n.ep > -2.3).all()
    assert (n.ep < 2.3).all()


n = place_electron_cloud_points(Nuke())
fig = eq.plot.Fig3D(cols=4, rows=3, scale=5)
ax = aks(n, fig.sub_plot_3d(title="electron"))
eq.plot.Animation(
    fig=fig.fig, func=animate, frames=4, interval=155, fargs=(n, ax)
).show()
