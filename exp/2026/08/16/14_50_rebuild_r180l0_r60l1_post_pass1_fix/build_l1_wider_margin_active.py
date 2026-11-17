"""R60 L0 / L1@R46.667 boundary case, "everything active" variant.

Same mesh/IC as exp/2026/08/11/15_05_l1_wider_margin/build_l0_l1_boundary.py
(margin=30 L1-cells past the spheromak edge -- the wider-margin case, not
the original margin=10), PLUS a seeded lam field:

    lam(x) = lam_scale * h(x),  h(x) = u(x)^2 + 0.5*sum_k phi(x,k)^2

matching build_uniform_r180_active.py exactly -- see that file's docstring
for the reasoning. cross_scale (step_cross) is applied in the step script,
not here.
"""

import argparse
import sys

sys.path.insert(0, ".")
import resource
import time

import numpy as np

from equilibrium import lattice as _lat
from run.substrate.adaptive.refined_neutron_decay import SPLIT_RATIO
from tst.substrate.adaptive import refine_region_to_convergence
from tst.substrate.adaptive.adaptive_flux_tube import _FACE_HOP, _FACE_NORM
from equilibrium.substrate.dipole_mode_ic import seed_dipole_ic

CORE_LEVEL = 1  # L1 is the finest level -- only one boundary to test
CELLS_ACROSS = 220

R_SPH_CENTRE = np.array([0.0, 0.0, 0.0])
R_SPH = 1.0
U_MAX, AMPLITUDE = 1.0, 1.0
BATCH = 500_000


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--r-l0",
        type=float,
        default=60.0,
        help="L0 background domain radius (absolute units)",
    )
    ap.add_argument(
        "--r-l1",
        type=float,
        default=40.0 + 20.0 / 3.0,
        help="radius refined to L1 -- wider-margin convention, see project_dipole_mode_ic",
    )
    ap.add_argument(
        "--lam-scale",
        type=float,
        default=1.0,
        help="lam = lam_scale * (u^2+0.5*sum(phi^2))",
    )
    ap.add_argument("--out", default="/bk/l1_wider_margin_active_ic220_pp1fix.h5")
    ap.add_argument("--n-threads", type=int, default=8)
    args = ap.parse_args()

    t0 = time.time()
    alat = _lat.AdaptiveLattice()
    alat.add_cluster()
    alat.set_n_threads(args.n_threads)
    cl = alat[0]
    _lat.build_bcc_sphere(cl, args.r_l0, d1=1.0, u0=0.0, ell0=1.0, lam0=0)
    alat.rebuild(0)
    print(
        f"[L0 base] R_L0={args.r_l0}  n_leaves={cl.n_leaves}  t={time.time()-t0:.1f}s  peak_rss_GB={rss_gb():.2f}",
        flush=True,
    )

    cl.info.split_hi = 0.5
    cl.info.split_lo = 0.0

    def region_fn(cl):
        pos = np.array(cl.leaf_pos)
        levels = np.array(cl.leaf_level)
        dist = np.linalg.norm(pos, axis=1)
        return (levels < CORE_LEVEL) & (dist <= args.r_l1)

    refine_region_to_convergence(alat, cl, region_fn, stage_max_level=CORE_LEVEL)
    N = cl.n_leaves
    levels = np.array(cl.leaf_level)
    print(
        f"[L1 core, r<={args.r_l1}] n_leaves={N}  level_hist={np.bincount(levels).tolist()}  "
        f"t={time.time()-t0:.1f}s  peak_rss_GB={rss_gb():.2f}",
        flush=True,
    )

    pos = np.array(cl.leaf_pos)
    cell_size = 1.0 / (SPLIT_RATIO**levels)

    SCALE = CELLS_ACROSS / (2 * SPLIT_RATIO**CORE_LEVEL)
    print(
        f"cells_across={CELLS_ACROSS}  scale={SCALE:.6f}  margin to R_L1={args.r_l1}: {args.r_l1-SCALE:.2f} "
        f"({(args.r_l1-SCALE)*SPLIT_RATIO**CORE_LEVEL:.1f} L1-cells)",
        flush=True,
    )

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
