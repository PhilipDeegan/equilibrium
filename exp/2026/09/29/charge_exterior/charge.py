"""Static charge exterior in the discrete step(): divergent phi core, uniform u.

phi_face = -W (psi_j - psi_i) with G^T G psi = s: div phi = s exactly, s = +Q at the core cell and
-Q spread over the outer shell (a closed lattice has zero net divergence: boundary faces carry
no phi, so the outer shell holds the image charge -- a spherical capacitor). u = 0 everywhere.

1  profile: psi(r) and |phi| vs the continuum Coulomb shape, direction dependence.
2  held: fixed_u on the core and shell cells -> must be exactly static (u_t = 0 everywhere,
   phi_t = -W grad u = 0).
3  not held: does the core divergence stay, or leave as a wave?
"""

import argparse
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from equilibrium.substrate.lattice import _K_REV, _W, SubstrateLattice, wave_speed

out = Path(__file__).parent
p = argparse.ArgumentParser()
p.add_argument("--R", type=float, default=12.0)
p.add_argument("--Q", type=float, default=1.0)
args = p.parse_args()

lat = SubstrateLattice(radius_cells=args.R)
N, nb = lat.N, lat.nb
r = np.linalg.norm(lat.pos, axis=1)
core = int(np.argmin(r))
shell = r > r.max() - 0.6

i_idx, k_idx = np.nonzero(nb >= 0)
j_idx = nb[i_idx, k_idx]
once = i_idx < j_idx
fi, fk, fj = i_idx[once], k_idx[once], j_idx[once]
F = len(fi)
w = _W[fk]
G = sp.csr_matrix(
    (np.concatenate([w, -w]), (np.concatenate([np.arange(F)] * 2), np.concatenate([fj, fi]))), shape=(F, N)
)
s = np.zeros(N)
s[core] = args.Q
s[shell] = -args.Q / shell.sum()
A = (G.T @ G).tocsr()
# fix psi on one shell cell (A is singular: constants)
pin = int(np.nonzero(shell)[0][0])
A = A.tolil()
A[pin, :] = 0
A[pin, pin] = 1.0
b = s.copy()
b[pin] = 0.0
psi = spsolve(A.tocsr(), b)
f = -(G @ psi)
phi = np.zeros((N, 14))
phi[fi, fk] = f
phi[fj, _K_REV[fk]] = -f


def div(ph):
    d = np.zeros(N)
    ii, kk = np.nonzero(nb >= 0)
    np.add.at(d, ii, _W[kk] * ph[ii, kk])
    return d


d0 = div(phi)
print(f"N={N} core div {d0[core]:.6f}  shell sum {d0[shell].sum():.6f}  max |div| elsewhere {np.abs(d0[~shell & (np.arange(N) != core)]).max():.2e}")

# 1 profile: psi - psi(shell) vs a (1/r - 1/b)
b_r = r[shell].mean()
m = (r > 1.5) & (r < b_r - 2.0)
X = 1.0 / r[m] - 1.0 / b_r
y = psi[m] - psi[shell].mean()
a = float(np.sum(X * y) / np.sum(X * X))
res = y - a * X
print(f"psi = a(1/r - 1/b): a = {a:.5f}  rel rms resid {np.sqrt(np.mean(res**2)) / np.abs(y).max():.2e}")
dirs = {"[100]": np.array([1, 0, 0.0]), "[110]": np.array([1, 1, 0.0]) / np.sqrt(2), "[111]": np.ones(3) / np.sqrt(3)}
for name, n in dirs.items():
    cosang = (lat.pos @ n) / np.maximum(r, 1e-12)
    sel = m & (cosang > 0.97)
    if sel.any():
        print(f"  {name}: psi*r/a at r~{r[sel].mean():.1f}: mean {np.mean((psi[sel] - psi[shell].mean()) / (a * (1 / r[sel] - 1 / b_r))):.4f}  (n={sel.sum()})")
# continuum: div(W grad) with the lattice's own isotropic coefficient eps = c^2 (lattice units):
# psi = Q / (4 pi eps_eff r) + const; eps_eff from the fit
print(f"eps_eff = Q/(4 pi a) = {args.Q / (4 * np.pi * a):.4f}   c^2 = {wave_speed() ** 2:.4f}   (cell volume 0.770)")

dt = lat.stable_dt()


def run(held, steps):
    L2 = SubstrateLattice(radius_cells=args.R)
    L2.phi[:] = phi
    L2.u[:] = 0.0
    if held:
        L2.fixed_u[core] = True
        L2.fixed_u[shell] = True
    hist = []
    for n in range(steps + 1):
        if n % max(1, steps // 10) == 0:
            dd = div(L2.phi)
            inner = (r < 3.0) & ~shell
            hist.append((n * dt, dd[core], dd[inner].sum(), np.abs(L2.u).max(), np.abs(L2.phi - phi).max()))
        L2.step(dt)
    return np.array(hist)


steps = int(round(3 * args.R / wave_speed() / dt))
for held in (True, False):
    h = run(held, steps)
    print(f"held={held}: t, core div, div in r<3, max|u|, max|phi - phi0|")
    for row in h:
        print("   " + "  ".join(f"{v:10.4e}" for v in row))
np.savez(out / "charge.npz", psi=psi, phi=phi, pos=lat.pos, core=core, shell=shell)
