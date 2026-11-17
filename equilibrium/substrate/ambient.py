"""
Ambient vacuum state for the step() rule, as an initial condition.

step() is linear and conservative with H = 1/2 sum_i u_i^2 + 1/2 sum_faces phi_f^2 (each face
stored on both cells, phi_ki = -phi_ik), and u, phi are its canonical coordinates. The thermal
(equipartition) ensemble at temperature T is therefore independent Gaussians,
u_i ~ N(0, T) and phi_f ~ N(0, T), and it is stationary: step() never moves energy between
normal modes, so no bath or noise is needed to hold it. <u^2> = T is exact, not fitted.

73.5% of the DOF (faces - cells + 1) are divergence-free phi, which step() leaves frozen: at T > 0
most of the energy sits there and cooling from outside cannot remove it
(exp/2026/09/29/ambient_cooling). T = 0 is the frozen floor state (uniform u, phi = 0).
"""

import numpy as np

K_REV = np.array([7, 6, 5, 4, 3, 2, 1, 0, 9, 8, 11, 10, 13, 12], dtype=np.intp)


def thermal_state(nb, T, rng=None, u_bg=0.0):
    """(u, phi) sampled from the equipartition ensemble at temperature T on a lattice with
    neighbour table nb (N, 14); phi antisymmetric, 0 on boundary faces."""
    rng = rng or np.random.default_rng()
    N = len(nb)
    u = u_bg + rng.normal(0.0, np.sqrt(T), N)
    phi = np.zeros((N, 14))
    i, k = np.nonzero(nb >= 0)
    j = nb[i, k]
    once = i < j
    i, k, j = i[once], k[once], j[once]
    g = rng.normal(0.0, np.sqrt(T), len(i))
    phi[i, k] = g
    phi[j, K_REV[k]] = -g
    return u, phi


def energy(u, phi, nb, u_bg=0.0):
    """H = 1/2 sum (u - u_bg)^2 + 1/4 sum over stored interior copies of phi^2."""
    return 0.5 * float(np.sum((u - u_bg) ** 2)) + 0.25 * float(np.sum(phi[nb >= 0] ** 2))
