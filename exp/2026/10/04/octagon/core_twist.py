"""internal state along a ring core: in 128 azimuth bins (about the core-weighted centre, z axis) take the core
centre (min rho cell); project the cell state on the two positive-frequency rest states v1 (seeded) and v2 (its
degenerate partner) and the negative-frequency pair; report core rho, |v1|^2, |v2|^2, |neg|^2 and the phase winding
of v1 and v2 along the loop. usage: core_twist.py state.npz [...]"""
import sys
from pathlib import Path

import numpy as np

from equilibrium.substrate.walk import symbol_dirac, to_complex

lam, V = np.linalg.eig(symbol_dirac(np.zeros(3), 0.8))
pos_f = np.flatnonzero(-np.angle(lam) > 0)
neg_f = np.flatnonzero(-np.angle(lam) < 0)
v1, v2 = V[:, pos_f[0]], V[:, pos_f[1]]


def wind(a):
    return np.sum(np.angle(np.roll(a, -1) / a)) / (2 * np.pi)


for f in map(Path, sys.argv[1:]):
    pos = np.load(f.parent / "pos.npy")
    ck = np.load(f)
    L, R = ck["L"], ck["R"]
    rho = (L**2).sum(1) + (R**2).sum(1)
    core = np.flatnonzero(rho < 0.25)
    c = pos[core].mean(0)
    q = pos[core] - c
    ph = np.arctan2(q[:, 1], q[:, 0])
    nb = 128
    k = ((ph + np.pi) / (2 * np.pi) * nb).astype(int) % nb
    idx = np.array([core[k == b][np.argmin(rho[core[k == b]])] for b in range(nb)])
    psi = np.concatenate([to_complex(L[idx]), to_complex(R[idx])], 1)
    a1, a2 = psi @ v1.conj(), psi @ v2.conj()
    an = psi @ V[:, neg_f].conj()
    print(f"{f.parent.name[10:22]} t {int(ck['t'])}: core-centre rho {rho[idx].mean():.4f} (max {rho[idx].max():.4f}); "
          f"|v1|^2 {np.mean(abs(a1)**2):.4f} |v2|^2 {np.mean(abs(a2)**2):.4f} |neg|^2 {np.mean((abs(an)**2).sum(1)):.4f}; "
          f"winding along loop v1 {wind(a1):+.2f} v2 {wind(a2):+.2f}; min |v2| {abs(a2).min():.3f}", flush=True)
