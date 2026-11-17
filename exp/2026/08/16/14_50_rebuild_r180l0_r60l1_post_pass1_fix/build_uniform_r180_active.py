"""Uniform L0 R180 baseline, "everything active" variant.

Same mesh/IC as exp/2026/08/11/15_05_l1_wider_margin/build_uniform_l0.py
(cells_across=220 fixes the spheromak radius at 110 absolute units,
--padding sets vacuum depth to the domain edge), PLUS a seeded lam field:

    lam(x) = lam_scale * h(x),  h(x) = u(x)^2 + 0.5*sum_k phi(x,k)^2

i.e. lam is proportional to the dipole mode's own local energy density at
t=0 -- a proxy for substrate depletion ("mc^2 removed from the substrate")
tied to wherever the object actually carries energy, zero in true vacuum.
No separate arbitrary radial profile. lam_scale=1.0 is an unvalidated first
guess (nothing in this codebase has calibrated lam for a dipole-mode IC
before) -- expect to revisit.

This is the no-junction reference for the "everything active" comparison
against build_l1_wider_margin_active.py; cross_scale (step_cross) is
applied identically in both cases' step scripts, not baked into the IC.
"""

import argparse
import sys

sys.path.insert(0, ".")
import resource
import time

import numpy as np

from equilibrium import lattice as _lat
from tst.substrate.adaptive.adaptive_flux_tube import _FACE_HOP, _FACE_NORM
from equilibrium.substrate.dipole_mode_ic import seed_dipole_ic

CELLS_ACROSS = 220  # spheromak's own point requirement -- always 220, fixed

R_SPH_CENTRE = np.array([0.0, 0.0, 0.0])
R_SPH = 1.0
U_MAX, AMPLITUDE = 1.0, 1.0
BATCH = 500_000


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--padding",
        type=float,
        default=70.0,
        help="decay-zone cells beyond spheromak edge -- 70 (R_domain=180) matches the boundary "
        "case's own L0 domain edge, see project_dipole_mode_ic",
    )
    ap.add_argument(
        "--lam-scale",
        type=float,
        default=1.0,
        help="lam = lam_scale * (u^2+0.5*sum(phi^2))",
    )
    ap.add_argument("--out", default="/bk/uniform_r180_active_ic220_pp1fix.h5")
    ap.add_argument("--n-threads", type=int, default=8)
    args = ap.parse_args()
    assert 20.0 <= args.padding <= 80.0, "padding must be 20-80 cells"

    SCALE = CELLS_ACROSS / 2.0  # radius in absolute units == L0-cell units, d1=1
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
        f"cells_across={CELLS_ACROSS}  scale={SCALE:.6f}  padding={args.padding:.1f} cells",
        flush=True,
    )

    pos = np.array(cl.leaf_pos)
    cell_size = np.full(N, 1.0)

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
