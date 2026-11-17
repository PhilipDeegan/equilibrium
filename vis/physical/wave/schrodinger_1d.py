# https://github.com/natsunoyuki/Computational_Physics_in_Python/blob/master/schrodinger1D.py
import argparse

import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sla


def harmonic(x, k=1.0):
    return k * x**2


def hamiltonian(x, V):
    n = len(x)
    dx = x[1] - x[0]
    T = sparse.diags_array([-1.0, 2.0, -1.0], offsets=[-1, 0, 1], shape=(n, n)) / dx**2
    return (T + sparse.diags_array(V)).tocsc()


def solve(xmin=-1, xmax=1, nx=5, neigs=3, V=harmonic):
    x = np.linspace(xmin, xmax, nx)
    evl, evt = sla.eigs(hamiltonian(x, V(x)), k=neigs, which="SM")
    for i in range(neigs):
        evt[:, i] /= np.sqrt(np.trapezoid(np.conj(evt[:, i]) * evt[:, i], x))
    order = np.argsort(np.real(evl))
    return np.real(evl)[order], evt[:, order], x


def figure(evl, evt, x):
    fig, axs = plt.subplots(len(evl), 1, figsize=(8, 8), squeeze=False)
    for i, ax in enumerate(axs[:, 0]):
        ax.plot(x, np.real(np.conj(evt[:, i]) * evt[:, i]))
        ax.axis("off")
    return fig


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--xmin", type=float, default=-1)
    ap.add_argument("--xmax", type=float, default=1)
    ap.add_argument("--nx", type=int, default=5, help="try 100 or 500")
    ap.add_argument("--neigs", type=int, default=3)
    a = ap.parse_args(argv)
    evl, evt, x = solve(a.xmin, a.xmax, a.nx, a.neigs)
    print("Energy eigenvalues:")
    for i, e in enumerate(evl):
        print(f"{i + 1}: {e:.2f}")
    figure(evl, evt, x)
    plt.show()


if __name__ == "__main__":
    main()
