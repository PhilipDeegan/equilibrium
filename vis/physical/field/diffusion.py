import dataclasses

import numpy as np

import equilibrium as eq


@dataclasses.dataclass
class Evolver:
    def __post_init__(self):
        self.A = np.zeros(100)
        X, W, off = 100, 10, 5
        self.A[np.linspace(off, X - off, W).astype(int)] = 100

    def __call__(self, i, mesh):
        R = 0.25
        X = 100
        A = self.A
        if i:
            t = A[:]
            for j in range(X - 1):  # periodic
                lt = (t[j - 1] - t[j]) * R
                A[j] += lt
                A[j - 1] -= lt
        mesh.set_array(A)


def animation():
    eq.plot.style.use("dark_background")

    X = 100
    N = 100

    e = Evolver()
    fig = eq.plot.Fig()
    ax = fig.sub_plot(
        ax_set={"xlim": [0, X], "ylim": [-0.5, 1.5], "xlabel": "dx", "ylabel": "dy"}
    )
    mesh = ax.pcolormesh(np.arange(0, X + 1), np.arange(0, 2), np.reshape(e.A, (1, X)))
    return eq.plot.Animation(fig=fig, func=lambda i: e(i, mesh), frames=N, interval=500)


def main():
    return animation().show()


if __name__ == "__main__":
    main()
