import numpy as np

from . import circle, point
from .line import Line

PI = np.longdouble("3.141592653589793238462643383279502884")


def lines(a, ps):
    return [Line(a, p) for p in ps]


def circle_points_rotate_y_axis(p, r, points=6):
    return circle.points(p, r, axis=2, points=points)


def line_closest_points(a, b, n=1):
    # count = len(a) if isinstance(a, list) else a.shape[0]
    vi = 0
    vs = [0] * n * a.shape[0]
    for a0 in a:
        d = ((a0 - b) ** 2).sum(axis=1)
        ndx = d.argsort()
        for f in b[ndx[:n]]:
            vs[vi] = Line(np.array([0] * 3), np.array([0] * 3))
            vs[vi].a = a0
            vs[vi].b = f
            vi += 1
    return vs


def sphere_points_y_axis_circles(p, r, points=3, N=2):
    if points < 3:
        raise ValueError("Sphere equator must be at least three points")

    mid = np.zeros([N * 2 + 1, points, 3])
    mid[N] = circle_points_rotate_y_axis(p, r, points)
    for i in range(N):
        x = np.cos(PI / 2 / (N + 1) * (i + 1)) * r
        y = np.sin(PI / 2 / (N + 1) * (i + 1)) * r
        mid[N - i - 1] = circle_points_rotate_y_axis(
            point.copy(p, y=p[1] + y), x, points
        )
        mid[N + i + 1] = circle_points_rotate_y_axis(
            point.copy(p, y=p[1] - y), x, points
        )

    return [point.copy(p, y=p[1] + r), mid, point.copy(p, y=p[1] - r)]


def sphere_vertices(p, r, points=3, N=0):
    """detail means how many vertices, 1 = as few as possible, so basically a diamond"""

    top, mid, bot = sphere_points_y_axis_circles(p, r, points, N)
    vs = lines(top, mid[0])
    for i in range(1, mid.shape[0]):
        vs += line_closest_points(mid[i - 1], mid[i])
    vs += lines(bot, mid[-1])
    return vs


class Sphere:
    def __init__(self, p, r, d=10, N=6):
        self.p = p
        self.r = r
        self.d = d
        self.N = N
        self._v = None

    def points(self):
        top, mid, bot = sphere_points_y_axis_circles(self.p, self.r, self.d, self.N)
        mid = mid.reshape([1, mid.shape[0] * mid.shape[1], mid.shape[2]])
        return np.concatenate(
            (top.reshape([1, 1, 3]), mid, bot.reshape([1, 1, 3])), axis=1
        )

    def vertices(self):
        return self.v

    @property
    def v(self):
        if not self._v:
            self._v = sphere_vertices(self.p, self.r, self.d, self.N)
        return self._v
