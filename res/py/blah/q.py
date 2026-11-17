import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse import linalg as sla


def schrodinger1D(xmin, xmax, Nx, Vfun, params, neigs=20):
    x = np.linspace(xmin, xmax, Nx)  # x axis grid
    dx = x[1] - x[0]  # x axis step size
    # Obtain the potential function values:
    V = Vfun(x, params)
    # create the Hamiltonian Operator matrix:
    H = sparse.eye(Nx, Nx, format="lil") * 2
    # implement the numerical derivative
    for i in range(Nx - 1):
        H[i, i + 1] = -1
        H[i + 1, i] = -1
    H = H / (dx**2)
    # Add in the potential energy V
    for i in range(Nx):
        H[i, i] = H[i, i] + V[i]
    # convert to csc sparse matrix format:
    H = H.tocsc()
    # obtain neigs solutions from the sparse matrix:
    [evl, evt] = sla.eigs(H, k=neigs, which="SM")
    for i in range(neigs):
        # normalize the eigenvectors:
        evt[:, i] = evt[:, i] / np.sqrt(np.trapezoid(np.conj(evt[:, i]) * evt[:, i], x))

    # eigen values MUST be real:
    evl = np.real(evl)

    return evl, evt, x


def eval_wavefunctions(xmin, xmax, Nx, Vfun, params, neigs):
    # call the 1D Schrodinger solver:
    evl, evt, x = schrodinger1D(xmin, xmax, Nx, Vfun, params, neigs)

    indices = np.argsort(evl)

    print("Energy eigenvalues:")
    for i, j in enumerate(evl[indices]):
        print("{}: {:.2f}".format(i + 1, j))

    i = 0
    plt.figure(figsize=(8, 8))
    while i < neigs:
        n = indices[i]
        # obtain probabilities from wave functions
        y = np.real(np.conj(evt[:, n]) * evt[:, n])
        # y = np.real(evt[:, n] ** 2)
        plt.subplot(neigs, 1, i + 1)
        plt.plot(x, y)
        plt.axis("off")
        i = i + 1
    plt.show()


def sho_wavefunctions_plot(xmin=-10, xmax=10, Nx=250, neigs=20, params=[1]):
    def Vfun(x, params):
        V = params[0] * x**2
        return V

    eval_wavefunctions(xmin, xmax, Nx, Vfun, params, neigs)


sho_wavefunctions_plot()
