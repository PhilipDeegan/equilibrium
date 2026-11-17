"""Continuum mutual flux for the gaussian tubes induction.py actually uses:
A_phi from the gaussian current (filament kernel integrated over (rho', z')),
then 2 pi a2 <A_phi> with induction.py's holonomy weight (gaussian around
loop 2, cell-count = volume weighting, i.e. x rho). mu0 = 1/C0SQ.
Also the same for loop 1's self-flux Phi1."""
import sys

import numpy as np
from scipy.special import ellipe, ellipk

C0SQ = 40.0 / 3.0
a1 = a2 = d = 2.0
s = float(sys.argv[1]) if len(sys.argv) > 1 else 0.4
n = 81
u = np.linspace(-4 * s, 4 * s, n)
du = u[1] - u[0]
G1 = np.exp(-0.5 * (u[:, None] ** 2 + u[None, :] ** 2) / s**2)


def A_phi(rho, z, a, zc):
    rp = a + u[:, None] + 0 * u[None, :]
    zp = zc + u[None, :] + 0 * u[:, None]
    J = G1 / (2 * np.pi * s**2)
    k2 = 4 * rp * rho / ((rp + rho) ** 2 + (z - zp) ** 2)
    k2 = np.clip(k2, 0, 1 - 1e-12)
    k = np.sqrt(k2)
    Af = (1 / C0SQ) / (np.pi * k) * np.sqrt(rp / rho) * ((1 - k2 / 2) * ellipk(k2) - ellipe(k2))
    return float((J * Af).sum() * du * du)


def flux(a, zc):
    num = den = 0.0
    for i, x in enumerate(u):
        for j, y in enumerate(u):
            w = G1[i, j] * (a + x)
            if w < 1e-8:
                continue
            num += w * A_phi(a + x, zc + y, a1, -d / 2)
            den += w
    return 2 * np.pi * a * num / den


k2 = 4 * a1 * a2 / ((a1 + a2) ** 2 + d**2)
k = np.sqrt(k2)
M = (1 / C0SQ) * np.sqrt(a1 * a2) * ((2 / k - k) * ellipk(k2) - (2 / k) * ellipe(k2))
print(f"s={s} Phi2/M (tube, continuum, unbounded) = {flux(a2, d/2)/M:.4f}  Phi1 = {flux(a1, -d/2):.4e}")
