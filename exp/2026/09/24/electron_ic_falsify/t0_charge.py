"""Falsification test T0 (static, t=0): does the electron IC carry net charge?

Working model (Philip 2026-09-24): the electron is a "glut" -- excess charge,
like static buildup. Charge = net outward flux through a closed surface
enclosing the IC (doc/dst.md: Q = sum_faces A_ij du/dr). An IC with zero net
flux through every enclosing sphere is neutral and is ruled out as an electron.

For each IC, sums across every face crossing the sphere |r|=R_s (inner leaf i
at |r_i|<R_s, neighbour j at |r_j|>=R_s):
  Q_u(R_s)   = -sum W_k (u_j - u_i)     (>0 for a u excess, outward gradient)
  Q_phi(R_s) =  sum W_k phi[k, i]       (net face flux out)
Also |Q| normalised by sum W_k |.| over the same faces, so "zero" is judged
relative to the field actually present on that shell, not in absolute units.
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/home/deegan/git/eq")
sys.path.insert(0, "/home/deegan/git/eq/exp/2026/09/22/dipole_ic_gallery")

import numpy as np

import plot_all_dipole_ics as gallery
from tst.substrate.adaptive.adaptive_flux_tube import (
    find_toroidal_ring,
    init_toroidal_winding,
    leaf_nb_j,
)

W_FACE = np.array([3.0] * 8 + [1.0] * 6)
R_SHELLS = [10.0, 20.0, 30.0, 40.0, 55.0, 65.0, 75.0]


def _seed_octagon_winding(cl, pos, cell_size):
    ring_js, ring_faces = find_toroidal_ring(cl, axis="z", mode="octagon", hops=12)
    init_toroidal_winding(cl, ring_js, ring_faces, amplitude=1.0)


PANELS = [(t.splitlines()[0], f) for t, f in gallery.PANELS] + [
    ("init_toroidal_winding (octagon hops=12)", _seed_octagon_winding)
]


def shell_charge(cl, nbj, r):
    u = np.asarray(cl.leaf_u, dtype=np.float64)
    phi = np.asarray(cl.leaf_phi, dtype=np.float64)
    out = {}
    for R_s in R_SHELLS:
        qu = qp = nu = npf = 0.0
        inside = r < R_s
        for k in range(14):
            j = nbj[:, k]
            m = inside & (j >= 0)
            m[m] &= r[j[m]] >= R_s
            i_idx = np.nonzero(m)[0]
            du = u[j[i_idx]] - u[i_idx]
            pk = phi[k, i_idx]
            qu += -W_FACE[k] * du.sum()
            nu += W_FACE[k] * np.abs(du).sum()
            qp += W_FACE[k] * pk.sum()
            npf += W_FACE[k] * np.abs(pk).sum()
        out[R_s] = (qu, qu / nu if nu else 0.0, qp, qp / npf if npf else 0.0)
    return out


def main():
    rows = []
    for name, seed_fn in PANELS:
        alat, cl = gallery.build_cluster()
        pos = np.array(cl.leaf_pos)
        r = np.linalg.norm(pos, axis=1)
        nbj = leaf_nb_j(cl)
        try:
            seed_fn(cl, pos, np.full(cl.n_leaves, 1.0))
        except Exception as exc:
            print(f"{name}: FAILED to seed: {exc}", flush=True)
            continue
        q = shell_charge(cl, nbj, r)
        print(f"\n=== {name}", flush=True)
        print(f"  {'R_s':>5} {'Q_u':>12} {'Q_u/|.|':>9} {'Q_phi':>12} {'Q_phi/|.|':>9}")
        for R_s, (qu, qun, qp, qpn) in q.items():
            print(f"  {R_s:5.0f} {qu:12.4e} {qun:9.2e} {qp:12.4e} {qpn:9.2e}", flush=True)
        rows.append((name, q))

    print("\n=== VERDICT (net charge on enclosing shells R_s=65,75; |rel|<1e-6 => neutral)")
    for name, q in rows:
        rel = max(abs(q[65.0][1]), abs(q[75.0][1]), abs(q[65.0][3]), abs(q[75.0][3]))
        absq = max(abs(q[65.0][0]), abs(q[75.0][0]), abs(q[65.0][2]), abs(q[75.0][2]))
        v = "NEUTRAL -> ruled out" if rel < 1e-6 or absq < 1e-10 else "CHARGED -> survives T0"
        print(f"  {name:60s} max|rel|={rel:.2e} max|Q|={absq:.2e}  {v}")


if __name__ == "__main__":
    main()
