import dataclasses

import numpy as np

import equilibrium as eq


@dataclasses.dataclass
class Evolver:
    def __post_init__(self):
        self.A = np.zeros([100, 100])
        X, Y, _N, _R, W, off = 100, 100, 100, 0.25, 25, 5
        x = np.linspace(off, X - off - 1, W).astype(int)
        y = np.linspace(off, Y - off - 1, W).astype(int)
        self.A[tuple(np.meshgrid(x, y))] = 100
        self.fn = self.first

    def first(self, frame, mesh):
        if frame:
            self.fn = self.rest
            self.fn(frame, mesh)
        mesh.set_array(self.A)

    def diff(self, a, b):
        R = 0.25
        d = a - b
        return d * R

    def rest(self, frame, mesh):
        R = 0.25
        X, Y = 100, 100
        A = self.A
        t = A[:]
        for y in range(Y):
            for x in range(X):
                lt = self.diff(t[y][x - 1], t[y][x])
                A[y][x] += lt
                A[y][x - 1] -= lt

                lt = self.diff(t[y - 1][x], t[y][x]) * R
                A[y][x] += lt
                A[y - 1][x] -= lt

                lt = self.diff(t[y - 1][x - 1], t[y - 1][x]) * R
                A[y][x] += lt
                A[y - 1][x - 1] -= lt
        np.testing.assert_allclose(A, t)
        mesh.set_array(A)

    def __call__(self, frame, mesh):
        self.fn(frame, mesh)


def animation():
    eq.plot.style.use("dark_background")

    X = 100
    Y = 100
    N = 100

    e = Evolver()
    fig = eq.plot.Fig(1, 1, 8)
    ax = fig.sub_plot(
        ax_set={"xlim": [0, X], "ylim": [0, Y], "xlabel": "dx", "ylabel": "dy"}
    )
    mesh = ax.pcolormesh(
        np.arange(0, X + 1), np.arange(0, Y + 1), np.reshape(e.A, (Y, X))
    )
    return eq.plot.Animation(fig=fig, func=lambda i: e(i, mesh), frames=N, interval=250)


def main():
    return animation().show()


if __name__ == "__main__":
    main()
