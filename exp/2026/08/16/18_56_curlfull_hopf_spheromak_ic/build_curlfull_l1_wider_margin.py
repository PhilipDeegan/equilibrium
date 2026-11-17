"""R60 L0 / L1@R46.667 boundary case, CURL-FULL variant -- companion to
build_l1_wider_margin_active.py (curl-free dipole_mode_ic). See
build_curlfull_uniform_r180.py's docstring for the shared reasoning
(project_dst_discrete_curl memory).

Ring-finding is restricted to LEVEL-1 leaves only (leaf_mask=levels==CORE_LEVEL)
via equilibrium.substrate.curl_full_spheromak_ic.find_square_ring_on_leaves --
find_toroidal_ring_at_level (adaptive_flux_tube.py) only supports the
minimal 8-tile ring, not the scaled "square" mode this needs.

hops=41 at L1 cell size (1/3) gives radius ~1.333*41*(1/3)~18.2, matching
0.5*R_SPH*SCALE = 0.5*36.667 = 18.33 for this config's SCALE -- the SAME
hops value as the uniform-R180 build works for both, since the ratio of
target-radius to cell-size cancels the SPLIT_RATIO exactly (both configs'
SCALE and cell-size scale together).
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
from equilibrium.substrate.curl_full_spheromak_ic import seed_curl_full_pair

CORE_LEVEL = 1
CELLS_ACROSS = 220


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r-l0", type=float, default=60.0)
    ap.add_argument("--r-l1", type=float, default=40.0 + 20.0 / 3.0)
    ap.add_argument("--hops", type=int, default=41)
    ap.add_argument("--amplitude-a", type=float, default=1.0)
    ap.add_argument("--amplitude-b", type=float, default=1.0)
    ap.add_argument("--lam-scale", type=float, default=1.0)
    ap.add_argument("--out", default="/bk/curlfull_l1_wider_margin_ic220.h5")
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

    SCALE = CELLS_ACROSS / (2 * SPLIT_RATIO**CORE_LEVEL)
    print(
        f"cells_across={CELLS_ACROSS}  scale={SCALE:.6f}  hops={args.hops}", flush=True
    )

    t1 = time.time()
    leaf_mask = levels == CORE_LEVEL
    res = seed_curl_full_pair(
        cl,
        hops=args.hops,
        amplitude_A=args.amplitude_a,
        amplitude_B=args.amplitude_b,
        leaf_mask=leaf_mask,
    )
    if res is None:
        print(
            "FAILED: could not find Hopf-linked ring pair at this hops/domain -- aborting",
            flush=True,
        )
        sys.exit(1)
    print(
        f"ring pair found: A={len(res['ring_A_js'])} tiles, B={len(res['ring_B_js'])} tiles  "
        f"t={time.time()-t1:.1f}s",
        flush=True,
    )

    u_rb = np.array(cl.leaf_u)
    phi_rb = np.array(cl.leaf_phi)
    h = u_rb**2 + 0.5 * np.sum(phi_rb**2, axis=0)
    lam = args.lam_scale * h
    cl.set_leaf_lam(lam)
    H0 = float(h.sum())
    print(
        f"H0={H0:.6e}  phi_rms={float(np.sqrt(np.mean(phi_rb**2))):.6e}  "
        f"lam_scale={args.lam_scale:g}  lam_max={float(lam.max()):.6e}  "
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
