"""Electron-candidate survival under self-sourced lam (Philip 2(a), 2026-09-29).

Discrete step() (sigma at the face on both legs), DampedLattice rate sponge. Every step:
    lam = eps + alpha * (u^2 + 1/2 sum_k phi_k^2)        (local field energy density x 2)
H = 1/2 sum u^2 + 1/2 sum_faces phi^2 is conserved for any sigma(t) with face sigma, so the
nonlinearity only redistributes energy. Divergence-free phi is static for any sigma (trivial
survival); the question is whether self-focusing (sigma drops where energy is high) traps the
propagating part. Metric: energy inside r < r_core vs time, against alpha = 0.

ICs: gauss (u bump), dipole (u = j1(k r)/(k r) * z/r-type l=1 bump), ring (numpy spheromak pair,
phi only: divergence-free control).
"""

import argparse
import itertools
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from equilibrium.substrate.flux_tube import find_toroidal_spheromak, init_spheromak
from equilibrium.substrate.lattice import DampedLattice, wave_speed

out = Path(__file__).parent
p = argparse.ArgumentParser()
p.add_argument("--R", type=float, default=12.0)
p.add_argument("--width", type=float, default=1.5)
p.add_argument("--crossings", type=float, default=6.0)
p.add_argument("--eps", type=float, default=0.0)
p.add_argument("--alphas", default="0,1,4")
p.add_argument("--amps", default="1,3")
p.add_argument("--ics", default="gauss,dipole,ring")
args = p.parse_args()

c = wave_speed()
r_core = 3.0 * args.width


def build(ic, amp):
    lat = DampedLattice(radius_cells=args.R, sponge_fraction=0.25)
    r = np.linalg.norm(lat.pos, axis=1)
    if ic == "gauss":
        lat.u[:] = amp * np.exp(-(r**2) / (2 * args.width**2))
    elif ic == "dipole":
        lat.u[:] = amp * (lat.pos[:, 2] / args.width) * np.exp(-(r**2) / (2 * args.width**2))
    elif ic == "ring":
        A, Af, B, Bf = find_toroidal_spheromak(lat, axis="z")
        init_spheromak(lat, A, Af, B, Bf, amplitude=amp, phi_only=True)
    return lat, r


def e_cell(lat):
    valid = lat.nb >= 0
    return 0.5 * lat.u**2 + 0.25 * np.sum(np.where(valid, lat.phi, 0.0) ** 2, axis=1)


import scipy.sparse as sp
from scipy.sparse.linalg import lsqr

from equilibrium.substrate.lattice import _W


def split_core(lat, core):
    """Core energy in u, gradient phi (range of W d) and divergence-free phi (the rest)."""
    nb = lat.nb
    i, k = np.nonzero(nb >= 0)
    j = nb[i, k]
    once = i < j
    fi, fk, fj = i[once], k[once], j[once]
    w = _W[fk]
    F = len(fi)
    G = sp.csr_matrix((np.concatenate([w, -w]), (np.concatenate([np.arange(F)] * 2), np.concatenate([fj, fi]))), shape=(F, lat.N))
    f = lat.phi[fi, fk]
    fg = G @ lsqr(G, f, atol=1e-12, btol=1e-12, iter_lim=20000)[0]
    fc = core[fi] & core[fj]
    return 0.5 * np.sum(lat.u[core] ** 2), 0.5 * np.sum(fg[fc] ** 2), 0.5 * np.sum((f - fg)[fc] ** 2)


rows, curves = [], {}
for ic, alpha, amp in itertools.product(
    args.ics.split(","), [float(a) for a in args.alphas.split(",")], [float(a) for a in args.amps.split(",")]
):
    lat, r = build(ic, amp)
    core = r < r_core
    dt = lat.stable_dt()
    n = int(round(args.crossings * 2 * args.R / c / dt))
    e0 = e_cell(lat)
    E0_core, E0 = e0[core].sum(), e0.sum()
    ts, ec = [], []
    for s in range(n):
        lat.lam[:] = args.eps + alpha * 2.0 * e_cell(lat)
        lat.step(dt)
        if s % 20 == 0 or s == n - 1:
            e = e_cell(lat)
            ts.append((s + 1) * dt)
            ec.append(e[core].sum() / E0_core)
    ts, ec = np.array(ts), np.array(ec)
    late = ec[ts > 0.7 * ts[-1]].mean()
    lam_max = float(lat.lam.max())
    rows.append((ic, alpha, amp, late, lam_max, e_cell(lat).sum() / E0))
    curves[(ic, alpha, amp)] = (ts, ec)
    eu, eg, es = split_core(lat, core)
    tot = eu + eg + es
    np.savez(out / f"end_{ic}_a{alpha:g}_A{amp:g}.npz", u=lat.u, phi=lat.phi, lam=lat.lam, pos=lat.pos)
    print(f"{ic:6s} alpha={alpha:4.1f} amp={amp:3.1f}: core energy late/initial {late:.3e}  "
          f"lam_max(end) {lam_max:.3e}  total left {e_cell(lat).sum() / E0:.3e}  core split u/grad/sol "
          f"{eu / tot:.3f}/{eg / tot:.3f}/{es / tot:.3f}", flush=True)

np.save(out / "self_lam_rows.npy", np.array(rows, dtype=object), allow_pickle=True)
ics = args.ics.split(",")
fig, ax = plt.subplots(1, len(ics), figsize=(6 * len(ics), 4.5), squeeze=False)
for a, ic in zip(ax[0], ics):
    for (k_ic, alpha, amp), (t, e) in curves.items():
        if k_ic == ic:
            a.semilogy(t, np.maximum(e, 1e-12), label=f"alpha={alpha:g} amp={amp:g}")
    a.set_title(f"{ic}: core energy (r<{r_core:g})")
    a.set_xlabel("t")
    a.legend(fontsize=7)
fig.tight_layout()
fig.savefig(out / "self_lam.png", dpi=110)
