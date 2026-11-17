"""Control test for the 08/11 everything_active divergence: a PURE single-level
mesh built directly at cell_size=1/3 (SPLIT_RATIO**-1) -- the exact absolute
cell size L1 cells have in build_l1_wider_margin_active.py -- with NO L0 shell,
no refinement, no AMR/junction machinery of any kind. Same physical domain
radius (60), same spheromak SCALE (36.667, cells_across=220 at this cell
size), same dipole IC, same lam seeding formula as both 08/11 active runs.

Purpose: isolate whether the observed H/H0 divergence (l1_wider_margin_active
climbing to 1.022 by step 130 while uniform_r180_active stays ~0.9995, even
though H_by_level shows the field has NOT reached the L0/L1 junction in bulk)
is caused by (a) the junction/proxy bookkeeping itself, active even before
bulk energy arrives there, or (b) something scale-dependent (cross_scale,
frozen lam, etc.) that isn't actually scale-covariant across absolute cell
size, independent of any AMR machinery. This mesh has NO junction at all --
if it reproduces the divergence, (b) is confirmed and the junction is
exonerated; if it stays flat like uniform_r180_active, (a) is implicated.
"""

import argparse
import sys

sys.path.insert(0, ".")
import resource
import time

import numpy as np

from equilibrium import lattice as _lat
from tst.substrate.adaptive.adaptive_flux_tube import _FACE_HOP, _FACE_NORM
from dipole_mode_ic import seed_dipole_ic

CELLS_ACROSS = 220
CELL_SIZE = (
    1.0 / 3.0
)  # matches L1's absolute cell size in build_l1_wider_margin_active.py

R_SPH_CENTRE = np.array([0.0, 0.0, 0.0])
R_SPH = 1.0
U_MAX, AMPLITUDE = 1.0, 1.0
BATCH = 500_000


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--r-domain",
        type=float,
        default=60.0,
        help="physical domain radius, matches R_L0 in the L1 case",
    )
    ap.add_argument(
        "--lam-scale",
        type=float,
        default=1.0,
        help="lam = lam_scale * (u^2+0.5*sum(phi^2))",
    )
    ap.add_argument("--out", default="/bk/uniform_l1size_pure_ic220.h5")
    ap.add_argument("--n-threads", type=int, default=8)
    args = ap.parse_args()

    SCALE = (
        CELLS_ACROSS / 2.0 * CELL_SIZE
    )  # == 36.667, matches build_l1_wider_margin_active.py exactly
    R_CELLS = args.r_domain / CELL_SIZE  # so physical radius == args.r_domain

    t0 = time.time()
    alat = _lat.AdaptiveLattice()
    alat.add_cluster()
    alat.set_n_threads(args.n_threads)
    cl = alat[0]
    _lat.build_bcc_sphere(cl, R_CELLS, d1=CELL_SIZE, u0=0.0, ell0=1.0, lam0=0)
    alat.rebuild(0)
    N = cl.n_leaves
    print(
        f"[pure L1-size] R_domain={args.r_domain:.3f} (R_cells={R_CELLS:.3f}, d1={CELL_SIZE:.6f})  "
        f"n_leaves={N}  t={time.time()-t0:.1f}s  peak_rss_GB={rss_gb():.2f}",
        flush=True,
    )
    print(f"cells_across={CELLS_ACROSS}  scale={SCALE:.6f}", flush=True)

    pos = np.array(cl.leaf_pos)
    cell_size = np.full(N, CELL_SIZE)

    r_cell = np.linalg.norm((pos - R_SPH_CENTRE) / SCALE, axis=1)
    n_inner = int((r_cell <= R_SPH).sum())
    print(f"inner_cells={n_inner}/{N} ({100*n_inner/N:.4f}%)", flush=True)

    seed_dipole_ic(
        cl,
        pos,
        cell_size,
        SCALE,
        _FACE_HOP,
        _FACE_NORM,
        centre=R_SPH_CENTRE,
        r_sph=R_SPH,
        u_max=U_MAX,
        amplitude=AMPLITUDE,
        batch=BATCH,
    )

    u_rb = np.array(cl.leaf_u)
    phi_rb = np.array(cl.leaf_phi)
    h = u_rb**2 + 0.5 * np.sum(phi_rb**2, axis=0)
    lam = args.lam_scale * h
    cl.set_leaf_lam(lam)
    H0 = float(h.sum())
    print(
        f"H0={H0:.6e}  phi_rms={float(np.sqrt(np.mean(phi_rb**2))):.6e}  "
        f"u_rms={float(np.sqrt(np.mean(u_rb**2))):.6e}  "
        f"lam_scale={args.lam_scale:g}  lam_max={float(lam.max()):.6e}  lam_mean={float(lam.mean()):.6e}  "
        f"peak_rss_GB={rss_gb():.2f}",
        flush=True,
    )

    t2 = time.time()
    alat.save(args.out)
    print(f"saved {args.out}  save_t={time.time()-t2:.1f}s", flush=True)
    print(
        f"FINAL total_t={time.time()-t0:.1f}s  peak_rss_GB={rss_gb():.2f}", flush=True
    )


if __name__ == "__main__":
    main()
