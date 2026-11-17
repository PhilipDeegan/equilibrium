"""Uniform L0 R180 baseline, using the NEW volume-filling curl-full spheromak
IC (equilibrium.substrate.curl_full_dipole_ic.seed_curl_full_dipole_ic) --
replaces both build_uniform_r180_active.py (curl-free dipole_mode_ic) and
build_curlfull_uniform_r180.py (bare wire-loop Hopf pair, u=0 everywhere,
~332/31M leaves) from 2026-08-16 for this config.

Same mesh/envelope as those two (cells_across=220, R_SPH=1, SCALE=110,
padding=70 -> R_domain=180), but u and phi now come from
seed_curl_full_dipole_ic: u/phi_pol identical to the dipole IC (two-lobed,
volume-filling, self-consistent scalar-wave-equation solution), PLUS a
genuinely-toroidal phi_tor = j1(K*r)*sin(theta) term (same K=4.493
eigenvalue) that carries real, nonzero macro-ring circulation -- see
equilibrium/substrate/curl_full_dipole_ic.py's docstring and
tst/substrate/adaptive/test_curl_full_dipole_ic.py for the validation this
construction is based on.
"""

import argparse
import sys

sys.path.insert(0, ".")
import resource
import time

import numpy as np

from equilibrium import lattice as _lat
from tst.substrate.adaptive.adaptive_flux_tube import _FACE_HOP, _FACE_NORM
from equilibrium.substrate.curl_full_dipole_ic import seed_curl_full_dipole_ic

CELLS_ACROSS = 220  # spheromak's own point requirement -- always 220, fixed

R_SPH_CENTRE = np.array([0.0, 0.0, 0.0])
R_SPH = 1.0
U_MAX = 1.0
AMPLITUDE_POL, AMPLITUDE_TOR = 1.0, 1.0
BATCH = 500_000


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--padding",
        type=float,
        default=70.0,
        help="decay-zone cells beyond spheromak edge -- 70 (R_domain=180) matches "
        "the boundary case's own L0 domain edge",
    )
    ap.add_argument(
        "--amplitude-tor",
        type=float,
        default=AMPLITUDE_TOR,
        help="peak |phi_tor| before summing with phi_pol",
    )
    ap.add_argument(
        "--lam-scale",
        type=float,
        default=1.0,
        help="lam = lam_scale * (u^2+0.5*sum(phi^2))",
    )
    ap.add_argument("--out", default="/bk/curlfull_dipole_uniform_r180_ic220.h5")
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

    u_all, phi_pol, phi_tor, phi_full = seed_curl_full_dipole_ic(
        cl,
        pos,
        cell_size,
        SCALE,
        _FACE_HOP,
        _FACE_NORM,
        centre=R_SPH_CENTRE,
        r_sph=R_SPH,
        u_max=U_MAX,
        amplitude_pol=AMPLITUDE_POL,
        amplitude_tor=args.amplitude_tor,
        batch=BATCH,
    )

    h = u_all**2 + 0.5 * np.sum(phi_full**2, axis=0)
    lam = args.lam_scale * h
    cl.set_leaf_lam(lam)
    H0 = float(h.sum())
    print(
        f"H0={H0:.6e}  phi_pol_rms={float(np.sqrt(np.mean(phi_pol**2))):.6e}  "
        f"phi_tor_rms={float(np.sqrt(np.mean(phi_tor**2))):.6e}  "
        f"u_rms={float(np.sqrt(np.mean(u_all**2))):.6e}  "
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
