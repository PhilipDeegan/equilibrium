"""Effective pi of lattice hop-path rings: path length / (2 * mean radius).
A hop path measures length in the lattice metric, so pi_eff need not -> pi
as the ring grows (staircase paradox)."""
import sys
sys.path.insert(0, "/home/deegan/git/eq")
import numpy as np
from equilibrium import lattice as _lat
from tst.substrate.adaptive.adaptive_flux_tube import find_toroidal_ring

alat = _lat.AdaptiveLattice(); alat.add_cluster(); cl = alat[0]
_lat.build_bcc_sphere(cl, 60.0, d1=1.0, u0=0.0, lam0=0.0); alat.rebuild(0)
pos = np.array(cl.leaf_pos)
for mode, hops_list in [("square", [None]), ("octagon", [2, 4, 8, 12, 16, 24])]:
    for h in hops_list:
        kw = {} if h is None else {"hops": h}
        js, _ = find_toroidal_ring(cl, axis="z", mode=mode, **kw)
        if js is None:
            print(mode, h, "not found"); continue
        P = pos[js]; c = P.mean(axis=0)
        L = np.linalg.norm(np.roll(P, -1, axis=0) - P, axis=1).sum()
        rho = np.hypot(P[:, 0] - c[0], P[:, 1] - c[1])
        print(f"{mode:8s} hops={h}  n={len(js):4d}  R_mean={rho.mean():7.3f}"
              f"  R_min/max={rho.min():.2f}/{rho.max():.2f}  pi_eff=L/2R={L/(2*rho.mean()):.5f}"
              f"  (pi={np.pi:.5f}, err={L/(2*rho.mean())/np.pi-1:+.3%})", flush=True)
