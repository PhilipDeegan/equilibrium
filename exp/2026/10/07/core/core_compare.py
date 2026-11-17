"""Relaxed coil core (10-06 branch/ profile_pp_N2_t8000.npz, core_profile.py: <rho> vs distance d to the rho < 0.02
cell set) against the exact GP q = 1 vortex (gp_profile_xi3.9.npz) sampled the SAME way: GP rho on a BCC lattice
(nn distance 1) around a straight line, core = cells with rho < 0.02, d = distance to the nearest core cell, 0.5 bins;
averaged over line orientations / offsets. -> where the 3 % (rho 0.5 at 5.11 vs 5.25) sits at all d."""
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

here = Path(__file__).parent
gp = np.load(here / "gp_profile_xi3.9.npz")
rel = np.load(here / "../../06/branch/profile_pp_N2_t8000.npz")
A = 2 / np.sqrt(3)
n = 40
g = np.arange(-n, n + 1) * A
X, Y, Z = np.meshgrid(g, g, g, indexing="ij")
pos = np.concatenate([np.stack([X, Y, Z], -1).reshape(-1, 3), np.stack([X, Y, Z], -1).reshape(-1, 3) + A / 2])
pos = pos[np.linalg.norm(pos, axis=1) < n * A - 2]
bins = np.arange(0, 30.5, 0.5)
dc = bins[:-1] + 0.25
rng = np.random.default_rng(0)
acc_s, acc_n = np.zeros(len(dc)), np.zeros(len(dc))
for _ in range(12):
    u = rng.normal(size=3)
    u /= np.linalg.norm(u)
    off = rng.uniform(-A, A, 3)
    q = pos - off
    rp = np.linalg.norm(q - np.outer(q @ u, u), axis=1)
    rho = np.interp(rp, gp["d"], gp["rho"])
    keep = np.abs(q @ u) < 25
    core = q[(rho < 0.02) & (np.abs(q @ u) < 32)]
    d = cKDTree(core).query(q[keep])[0]
    k = np.digitize(d, bins) - 1
    ok = k < len(dc)
    acc_s += np.bincount(k[ok], rho[keep][ok], minlength=len(dc))
    acc_n += np.bincount(k[ok], minlength=len(dc))
prof = acc_s / np.maximum(acc_n, 1)
rr = np.interp(dc, rel["d"], rel["rho"], right=np.nan)
print("  d      GP(lattice, same metric)  relaxed    rel - GP")
for j in range(0, len(dc), 2):
    if acc_n[j] and not np.isnan(rr[j]):
        print(f"  {dc[j]:5.2f}  {prof[j]:.4f}                    {rr[j]:.4f}     {rr[j] - prof[j]:+.4f}")
m = acc_n > 0
print(f"rho 0.5 at d: GP-lattice {np.interp(0.5, prof[m], dc[m]):.2f}, relaxed {np.interp(0.5, rel['rho'], rel['d']):.2f}")
for lv in (0.2, 0.8, 0.9, 0.95):
    print(f"rho {lv} at d: GP-lattice {np.interp(lv, prof[m], dc[m]):.2f}, relaxed {np.interp(lv, rel['rho'], rel['d']):.2f}")
