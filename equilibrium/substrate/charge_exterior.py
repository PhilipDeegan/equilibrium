"""
Static charge exterior for the step() rule: a divergent face field with uniform u.

step() (u_t = -sum_k W_k phi_k, phi_t = -W_k (u_nb - u)) is static exactly when u is uniform and
div phi = 0. A charge's exterior is therefore radial phi, not a u gradient: phi_face =
-W (psi_j - psi_i) with the weighted Laplacian G^T G psi = s puts div phi = s exactly. A closed
lattice has zero net divergence (boundary faces carry no phi), so +Q at a core needs -Q on the
outer shell: a discrete spherical capacitor, psi = Q / (4 pi eps) (1/r - 1/b) with
eps = c^2 / V_cell (exp/2026/09/29/charge_exterior/status.md).

What holds the core's divergence is the open charge question; holding u there at the background
(fixed_u) stands in for it. A bare core does not hold: the divergence leaves as u waves.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

K_REV = np.array([7, 6, 5, 4, 3, 2, 1, 0, 9, 8, 11, 10, 13, 12], dtype=np.intp)


def divergence(phi, nb, W):
    """sum_k W_k phi_ik over interior faces, per cell."""
    i, k = np.nonzero(nb >= 0)
    d = np.zeros(len(nb))
    np.add.at(d, i, W[k] * phi[i, k])
    return d


def seed_charge(nb, W, source):
    """
    Face field phi (N, 14), antisymmetric, 0 on boundary faces, with divergence(phi) == source.
    source must sum to zero (closed lattice). Returns (phi, psi).
    """
    source = np.asarray(source, dtype=np.float64)
    if abs(source.sum()) > 1e-12 * max(np.abs(source).max(), 1.0):
        raise ValueError("seed_charge: source must sum to zero on a closed lattice")
    N = len(nb)
    i, k = np.nonzero(nb >= 0)
    j = nb[i, k]
    once = i < j
    fi, fk, fj = i[once], k[once], j[once]
    w = W[fk]
    F = len(fi)
    G = sp.csr_matrix(
        (np.concatenate([w, -w]), (np.concatenate([np.arange(F)] * 2), np.concatenate([fj, fi]))), shape=(F, N)
    )
    A = (G.T @ G).tolil()
    pin = int(np.argmin(source))
    A[pin, :] = 0.0
    A[pin, pin] = 1.0
    b = source.copy()
    b[pin] = 0.0
    psi = spsolve(A.tocsr(), b)
    f = -(G @ psi)
    phi = np.zeros((N, 14))
    phi[fi, fk] = f
    phi[fj, K_REV[fk]] = -f
    return phi, psi


def capacitor_source(pos, Q=1.0, shell_depth=0.6):
    """+Q on the cell nearest the origin, -Q spread over cells within shell_depth of the edge.
    Returns (source, core index, shell mask)."""
    r = np.linalg.norm(pos, axis=1)
    core = int(np.argmin(r))
    shell = r > r.max() - shell_depth
    s = np.zeros(len(pos))
    s[core] = Q
    s[shell] = -Q / shell.sum()
    return s, core, shell
