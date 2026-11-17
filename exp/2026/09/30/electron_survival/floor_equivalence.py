"""Uniform additive floor eps vs void: lam = eps + alpha 2e with face sigma = mean cell sigma.
sigma = s0 / (1 + s0 alpha 2e), s0 = 1/(1+eps) -> (eps, alpha, dt) == (0, s0 alpha, s0 dt) exactly
(no sponge here: plain lattice, the rate sponge is in coordinate time and would break it)."""
import numpy as np
from equilibrium.substrate.lattice import SubstrateLattice

def e_cell(lat):
    return 0.5 * lat.u**2 + 0.25 * np.sum(np.where(lat.nb >= 0, lat.phi, 0.0) ** 2, axis=1)

def run(eps, alpha, dt, n):
    lat = SubstrateLattice(radius_cells=8.0)
    r = np.linalg.norm(lat.pos, axis=1)
    lat.u[:] = np.exp(-r**2 / (2 * 1.5**2))
    for _ in range(n):
        lat.lam[:] = eps + alpha * 2.0 * e_cell(lat)
        lat.step(dt)
    return lat.u.copy()

eps, alpha = 0.25, 36.0
s0 = 1 / (1 + eps)
dt = SubstrateLattice(radius_cells=8.0).stable_dt()
a = run(eps, alpha, dt, 400)
b = run(0.0, s0 * alpha, s0 * dt, 400)
print(f"max|u_floor - u_void(rescaled)| = {np.abs(a - b).max():.3e}   max|u| = {np.abs(a).max():.3e}")
