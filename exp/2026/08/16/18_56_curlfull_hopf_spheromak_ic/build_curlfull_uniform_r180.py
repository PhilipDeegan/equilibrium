"""Uniform L0 R180 baseline, CURL-FULL variant -- companion to
build_uniform_r180_active.py (which uses the curl-free dipole_mode_ic), for
the project_dst_discrete_curl investigation into PHARE-style flux-tube
merging.

Seeds a genuinely curl-full Hopf-linked ring pair
(equilibrium.substrate.curl_full_spheromak_ic.seed_curl_full_pair) instead
of dipole_mode_ic's phi=grad(U): Ring A (a `hops`-scaled toroidal loop) +
Ring B (a small linked plaquette), BOTH given a CONSTANT phi value around
the loop (not a standing-wave cos pattern -- see module docstring for why
that distinction matters). Validated at hops=41 on a smaller test domain
(project_dst_discrete_curl memory): both rings' circulation exactly
conserved under pure step(), erodes under step_cross.

hops=41 chosen so Ring A's physical radius sits at ~half of R_SPH*SCALE
(the same envelope build_uniform_r180_active.py's dipole IC uses) --
empirically radius=1.333*hops*cell_size (d1=1 here), giving radius~54.6 vs
target 0.5*110=55.

IMPORTANT physical difference from the dipole IC: u is NOT seeded here (left
at 0 everywhere) -- ALL the field content lives on the ring/plaquette faces
themselves (~332 leaves out of 30M+), not spread through a smooth spherical
volume. lam = lam_scale*(u^2+0.5*sum(phi^2)) is therefore a thin ring/tube
shape, not a smooth spheromak-volume profile like the dipole IC's. This is
expected for a genuine current-loop construction, not a bug -- flagged
explicitly, not hidden.
"""

import argparse
import sys

sys.path.insert(0, ".")
import resource
import time

import numpy as np

from equilibrium import lattice as _lat
from equilibrium.substrate.curl_full_spheromak_ic import seed_curl_full_pair

CELLS_ACROSS = 220


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--padding",
        type=float,
        default=70.0,
        help="decay-zone cells beyond R_SPH -- matches build_uniform_r180_active.py",
    )
    ap.add_argument(
        "--hops",
        type=int,
        default=41,
        help="Ring A square-loop hops -- radius ~1.333*hops*d1",
    )
    ap.add_argument("--amplitude-a", type=float, default=1.0)
    ap.add_argument("--amplitude-b", type=float, default=1.0)
    ap.add_argument(
        "--lam-scale",
        type=float,
        default=1.0,
        help="lam = lam_scale * (u^2+0.5*sum(phi^2))",
    )
    ap.add_argument("--out", default="/bk/curlfull_uniform_r180_ic220.h5")
    ap.add_argument("--n-threads", type=int, default=8)
    args = ap.parse_args()

    SCALE = (
        CELLS_ACROSS / 2.0
    )  # == R_SPH*SCALE with R_SPH=1.0 -- same envelope as the dipole IC
    R_DOMAIN = SCALE + args.padding

    t0 = time.time()
    alat = _lat.AdaptiveLattice()
    alat.add_cluster()
    alat.set_n_threads(args.n_threads)
    cl = alat[0]
    _lat.build_bcc_sphere(cl, R_DOMAIN, d1=1.0, u0=0.0, ell0=1.0, lam0=0)
    alat.rebuild(0)
    N = cl.n_leaves
    print(
        f"[uniform L0] R_domain={R_DOMAIN:.3f}  n_leaves={N}  t={time.time()-t0:.1f}s  peak_rss_GB={rss_gb():.2f}",
        flush=True,
    )
    print(
        f"cells_across={CELLS_ACROSS}  scale={SCALE:.6f}  padding={args.padding:.1f}  hops={args.hops}",
        flush=True,
    )

    t1 = time.time()
    res = seed_curl_full_pair(
        cl, hops=args.hops, amplitude_A=args.amplitude_a, amplitude_B=args.amplitude_b
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
