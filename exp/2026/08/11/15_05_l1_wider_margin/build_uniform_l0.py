"""Uniform L0 spheromak baseline -- NO refinement anywhere, cell size = 1
everywhere. "Level" is meaningless for a whole-domain build (levels only
exist relative to L0), so there is no --level knob here on purpose.

cells_across=220 fixes the spheromak radius at 110 absolute (=L0-cell)
units; --padding (in cells, 20-40 per spec) sets how much vacuum lies
between the spheromak's edge and the domain boundary.
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
        help="decay-zone cells beyond spheromak edge -- 70 (R_domain=180) here so the reference "
        "domain edge (r/Rsph=1.636) matches the boundary case's own L0 domain edge "
        "(R_L0=60/Rsph=36.667), rather than the original 20-40 spec range which put the domain "
        "edge (r/Rsph=1.273) right on top of the margin=30 L1 junction -- see project_dipole_mode_ic",
    )
    ap.add_argument("--out", default="/bk/uniform_l0_r180_ic220.h5")
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
    H0 = float(np.sum(u_rb**2) + 0.5 * np.sum(phi_rb**2))
    print(
        f"H0={H0:.6e}  phi_rms={float(np.sqrt(np.mean(phi_rb**2))):.6e}  "
        f"u_rms={float(np.sqrt(np.mean(u_rb**2))):.6e}  peak_rss_GB={rss_gb():.2f}",
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
