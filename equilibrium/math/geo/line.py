import numpy as np

import equilibrium as eq


class Line:
    def __init__(self, a: np.ndarray, b):
        self.a = eq.math.numpy.to_array(a)
        self.b = eq.math.numpy.to_array(b)
        if self.a.shape != self.b.shape:
            raise ValueError("Line points shape mismatch")
        s = eq.math.numpy.to_array(self.a.shape)
        if len(s) == 0 or s != s.flatten():
            raise ValueError("Line point must be 1d")

    def xyz(self):
        a, b = self.a, self.b
        return ((a[i], b[i]) for i in range(self.a.shape[0]))

    def vector(self):
        return self.b - self.a

    def vector_pos(self):
        return self.a, self.vector()

    def mid(self):
        a, b = self.a, self.b
        return ((a[i] + b[i] / 2) for i in range(self.a.shape[0]))

    def __contains__(self, p):
        a, b = self.a, self.b
        return (b >= p).all() and (p >= a).all()

    def __repr__(self):
        return f"Line({self.a}, {self.b})"

    @staticmethod
    def zero_to(x):
        return Line(np.zeros_like(x), x)


def cross(a, b):
    return np.cross(a.vector(), b.vector())


def dot(a, b):
    return np.dot(a.vector(), b.vector())


def inner(a, b):
    return np.inner(a.vector(), b.vector())


class Vector:
    def __init__(self, c: np.ndarray):
        self.c = c


def intersection(a, b):
    """
    Returns point of intersection, or none
    """

    c = a.b - a.a
    d = b.b - b.a
    f = np.cross(c, d)
    g = np.cross(c, f)
    h = np.dot(g, b.a)
    i = np.dot(g, d)
    j = np.dot(g, a.a)
    k = (h - j) / i * -1

    p = b.a + np.dot(d, k)
    return p if p in a and p in b else None


def intersects(a, b):
    return intersection(a, b) is not None
