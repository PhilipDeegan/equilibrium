"""
Charge test 1: discrete Gauss's law on the one IC in this codebase that
actually sources a nonuniform `u` field.

Context (2026-08-27 session): doc/dst.md defines charge as
  Q_i = Sum_k A_ik (u_nb_k - u_i) / d_ik
purely in terms of the `u` field's discrete Laplacian -- the natural
Gaussian surface being the cell's own 14 faces. But neither of the two
objects used all session (A2's `set_winding`-built "proton knot": only
touches `lam`, u stays 0; `init_electron_dipole_ic`: explicitly
"zero-net-force construction: u is set UNIFORM") ever sources a nonzero,
nonuniform u -- so Q_i=0 for both, structurally, by construction, and no
pairwise force test done today could ever have found a Coulomb-like signal
regardless of gate/mechanism.

`init_spheromak_ic` (the "neutron" IC, used throughout the R1/R2 gate work)
is the one exception: it sets `u = |B|` at cell centre from the Taylor
spheromak's own field. This script checks whether that field actually
behaves like DST's own formal definition of a charge:
  1. Q_i per cell -- should be ~0 away from the knot (doc's own claim:
     "Free cell: Q_i=0 by equilibrium"), nonzero only near the source.
  2. Q_enc(r) = Sum_{r'<r} Q_i -- should be constant once the shell fully
     encloses the source region (the actual, non-trivial Gauss's law check;
     total enclosed charge must not depend on how big a shell you draw).
  3. Caveat to watch for: u=|B| is a magnitude (always >= 0) -- may not
     carry the sign structure a real charge needs. Report Q_i's own sign
     distribution, don't assume.

Uses the SAME weights/distances the real dynamics use (_W_FACE=[3]*8+[1]*6,
matching doc's 2:1 hex:sq ratio; _D_FACE=[1.0]*8+[2/sqrt(3)]*6, the same
axial-spacing fact already established this session for `equilibrium.
substrate.lattice`), applied to the AdaptiveCellCluster leaf arrays.

Run:
  PYTHONPATH=$PWD python3 -u exp/2026/08/27/19_25_gauss_charge_tests/probe_gauss_charge.py
"""

import sys

import numpy as np

sys.path.insert(0, ".")
from equilibrium import lattice as _lat
from run.substrate.adaptive.cross_v2 import _W_FACE
from tst.substrate.adaptive.adaptive_flux_tube import init_spheromak_ic, leaf_nb_j

_D_FACE = np.array([1.0] * 8 + [2.0 / 3.0**0.5] * 6)


def main():
    R, sph_frac, B0 = 30.0, 0.2, 1.0
    scale = R * sph_frac

    alat = _lat.AdaptiveLattice()
    alat.add_cluster()
    alat.set_n_threads(8)
    cl = alat[0]
    _lat.build_bcc_sphere(cl, R, d1=1.0, u0=0.0, ell0=1.0, lam0=0)
    alat.rebuild(0)

    init_spheromak_ic(
        cl, R_sph=1.0, scale=scale, B0=B0, lam_target=0.1, u_max=1.0, amplitude=0.0
    )

    pos = np.array(cl.leaf_pos)
    r = np.linalg.norm(pos, axis=1)
    u = np.array(cl.leaf_u)
    nb = leaf_nb_j(cl)
    valid = nb >= 0
    safe_nb = np.where(valid, nb, 0)

    u_nb = np.where(valid, u[safe_nb], u[:, None])
    du = u_nb - u[:, None]
    Q_i = np.where(valid, _W_FACE[None, :] * du / _D_FACE[None, :], 0.0).sum(axis=1)

    print(f"N={cl.n_leaves}  R={R}  scale={scale:.1f}")
    print(
        f"u: min={u.min():.4e}  max={u.max():.4e}  (u=|B|, always >=0 by construction)"
    )
    print(f"Q_i: min={Q_i.min():.4e}  max={Q_i.max():.4e}  mean={Q_i.mean():.4e}")
    print(
        f"Q_i sign split: {(Q_i > 1e-12).sum()} positive, {(Q_i < -1e-12).sum()} negative,"
        f" {(np.abs(Q_i) <= 1e-12).sum()} ~zero  (of {cl.n_leaves})"
    )

    # Q_i should be near-zero far from the knot -- check it directly
    far_mask = r > scale * 3.0
    print(
        f"\nfar-field (r > 3*scale = {3*scale:.1f}) |Q_i|: "
        f"mean={np.abs(Q_i[far_mask]).mean():.4e}  max={np.abs(Q_i[far_mask]).max():.4e}"
        f"  (n={int(far_mask.sum())} cells)"
    )

    # Q_enc(r): cumulative enclosed "charge" as a function of shell radius
    r_bins = np.linspace(0, R * 0.9, 40)
    print(f"\n{'r':>8}  {'Q_enc(r)':>12}  {'n_enclosed':>10}")
    for rb in r_bins[::2]:
        mask = r < rb
        print(f"{rb:8.2f}  {Q_i[mask].sum():12.4e}  {int(mask.sum()):10d}")


if __name__ == "__main__":
    main()
