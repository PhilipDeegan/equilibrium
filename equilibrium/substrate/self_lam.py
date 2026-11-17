"""Self-sourced lam on the discrete step(): lam = eps + alpha (u^2 + 1/2 sum_k phi_k^2), sigma = 1/(1+lam).

Implicit midpoint (sigma at the midpoint) conserves H exactly; symplectic Euler has an O(dt) energy error once sigma
depends on the state (exp/2026/09/30/electron_survival). `step_midpoint_numpy` is the reference (from
self_lam_long.py); `SelfLamMidpoint` runs the same step in C++ (eq::substrate_step_selflam_midpoint), threaded,
double-buffered so no per-step copies.
"""

import numpy as np

from equilibrium.substrate.lattice import _W


def energy_cell(lat, u=None, phi=None):
    u = lat.u if u is None else u
    phi = lat.phi if phi is None else phi
    return 0.5 * u**2 + 0.25 * np.sum(np.where(lat.nb >= 0, phi, 0.0) ** 2, axis=1)


def rhs(lat, alpha, u, phi, eps=0.0):
    valid = lat.nb >= 0
    nb = np.where(valid, lat.nb, 0)
    lam = eps + alpha * (u**2 + 0.5 * np.sum(np.where(valid, phi, 0.0) ** 2, axis=1))
    sig = 1.0 / (1.0 + lam)
    sw = np.where(valid, 0.5 * (sig[:, None] + sig[nb]) * _W[None, :], 0.0)
    return -(sw * phi).sum(axis=1), -sw * (u[nb] - u[:, None]), lam


def step_midpoint_numpy(lat, alpha, dt, eps=0.0, tol=1e-13, max_iter=200):
    """x1 = x0 + dt f(x_mid); sets lat.u, lat.phi, lat.lam (lam at the last midpoint); returns iterations."""
    u0, p0 = lat.u, lat.phi
    du, dp, _ = rhs(lat, alpha, u0, p0, eps)
    u1, p1 = u0 + dt * du, p0 + dt * dp
    for it in range(max_iter):
        du, dp, lam = rhs(lat, alpha, 0.5 * (u0 + u1), 0.5 * (p0 + p1), eps)
        un, pn = u0 + dt * du, p0 + dt * dp
        err = max(np.abs(un - u1).max(), np.abs(pn - p1).max())
        u1, p1 = un, pn
        if err < tol:
            break
    else:
        raise RuntimeError(f"midpoint did not converge: {err:.2e}")
    lat.u, lat.phi, lat.lam[:] = u1, p1, lam
    return it + 1


class SelfLamMidpoint:
    """C++ implicit-midpoint stepper bound to a lattice; step() swaps lat.u / lat.phi with internal buffers."""

    def __init__(self, lat, alpha, eps=0.0, tol=1e-13, max_iter=200, n_threads=1):
        from equilibrium import cpp

        self._step = cpp.substrate_step_selflam_midpoint
        self.lat, self.alpha, self.eps, self.tol, self.max_iter, self.n_threads = lat, alpha, eps, tol, max_iter, n_threads
        self.nb = np.ascontiguousarray(lat.nb, dtype=np.int32)
        self._u = np.empty(lat.N)
        self._phi = np.empty((lat.N, 14))
        self._um = np.empty(lat.N)
        self._sig = np.empty(lat.N)
        self.lam = np.zeros(lat.N)

    def step(self, dt):
        lat = self.lat
        u0 = np.ascontiguousarray(lat.u, dtype=np.float64)
        p0 = np.ascontiguousarray(lat.phi, dtype=np.float64)
        it = self._step(u0, p0, self._u, self._phi, self.lam, self._um, self._sig, self.nb, dt, self.alpha, self.eps,
                        self.tol, self.max_iter, self.n_threads)
        lat.u, lat.phi, self._u, self._phi = self._u, self._phi, u0, p0
        lat.lam[:] = self.lam
        return it
