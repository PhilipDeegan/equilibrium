"""Cool a thermal ambient state through a boundary sponge: what can't be removed?

Discrete step() (lam = 0, sigma = 1). H = 1/2 sum u^2 + 1/4 sum_i sum_k phi_ik^2 (each face
stored on both cells, phi_ki = -phi_ik). Thermal IC: u_i ~ N(0, T), phi_face ~ N(0, T)
(equipartition over canonical coordinates). phi splits into gradient G psi,
(G psi)_ik = W_k (psi_k - psi_i), and the weighted-divergence-free rest, which step() leaves
frozen when u is flat. Sponge (DampedLattice) damps u and phi in the outer shell.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr

from equilibrium.substrate.lattice import _K_REV, _W, DampedLattice

p = argparse.ArgumentParser()
p.add_argument("--R", type=float, default=12.0)
p.add_argument("--T", type=float, default=1.0)
p.add_argument("--steps", type=int, default=40000)
p.add_argument("--every", type=int, default=400)
p.add_argument("--damp", type=float, default=0.02)
p.add_argument("--sponge", type=float, default=0.2)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", default="cool_R12")
args = p.parse_args()

out = Path(__file__).parent
lat = DampedLattice(radius_cells=args.R, sponge_fraction=args.sponge, damp_max=args.damp)
N = lat.N
nb = lat.nb
r = np.linalg.norm(lat.pos, axis=1)
interior = lat._sponge == 0.0

i_idx, k_idx = np.nonzero(nb >= 0)
j_idx = nb[i_idx, k_idx]
once = i_idx < j_idx
fi, fk, fj = i_idx[once], k_idx[once], j_idx[once]
F = len(fi)
face_interior = interior[fi] & interior[fj]

rng = np.random.default_rng(args.seed)
lat.u[:] = rng.normal(0.0, np.sqrt(args.T), N)
g = rng.normal(0.0, np.sqrt(args.T), F)
lat.phi[:] = 0.0
lat.phi[fi, fk] = g
lat.phi[fj, _K_REV[fk]] = -g

w = _W[fk]
G = sp.csr_matrix(
    (np.concatenate([w, -w]), (np.concatenate([np.arange(F)] * 2), np.concatenate([fj, fi]))),
    shape=(F, N),
)
print(f"N={N} faces={F} frozen dim={F - N + 1} ({(F - N + 1) / (F + N):.3f} of all DOF)", flush=True)


def split():
    f = lat.phi[fi, fk]
    psi = lsqr(G, f, atol=1e-12, btol=1e-12, iter_lim=20000)[0]
    fg = G @ psi
    fs = f - fg
    return f, fg, fs


def energies():
    f, fg, fs = split()
    return dict(
        u=0.5 * np.sum(lat.u**2),
        grad=0.5 * np.sum(fg**2),
        sol=0.5 * np.sum(fs**2),
        u_in=0.5 * np.sum(lat.u[interior] ** 2),
        grad_in=0.5 * np.sum(fg[face_interior] ** 2),
        sol_in=0.5 * np.sum(fs[face_interior] ** 2),
        antisym=float(np.max(np.abs(lat.phi[fi, fk] + lat.phi[fj, _K_REV[fk]]))),
    )


dt = lat.stable_dt()
rec = []
e = energies()
rec.append((0, e))
print(0, {k: round(v, 3) for k, v in e.items()}, flush=True)
for s in range(1, args.steps + 1):
    lat.step(dt)
    if s % args.every == 0:
        e = energies()
        rec.append((s, e))
        if s % (args.every * 10) == 0:
            print(s, {k: round(v, 3) for k, v in e.items()}, flush=True)

steps = np.array([s for s, _ in rec])
keys = list(rec[0][1])
E = {k: np.array([e[k] for _, e in rec]) for k in keys}
np.savez(
    out / f"{args.out}.npz",
    t=steps * dt,
    **{f"E_{k}": v for k, v in E.items()},
    u=lat.u,
    phi=lat.phi,
    pos=lat.pos,
    interior=interior,
    args=str(vars(args)),
)

t = steps * dt
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
for k in ("u", "grad", "sol"):
    ax[0].plot(t, E[k] / E[k][0], label=k)
    ax[1].loglog(t[1:], E[k][1:] / E[k][0], label=k)
for a in ax:
    a.set_xlabel("t")
    a.set_ylabel("E / E(0)")
    a.legend()
ax[0].set_title(f"R={args.R} damp={args.damp} sponge={args.sponge}")
fig.tight_layout()
fig.savefig(out / f"{args.out}.png", dpi=110)
print("saved", out / f"{args.out}.npz")
