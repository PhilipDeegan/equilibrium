import argparse
import sys

sys.path.insert(0, ".")
import resource
import time

import numpy as np
from scipy.special import spherical_jn

from equilibrium import lattice as _lat
from run.substrate.adaptive.refined_neutron_decay import SPLIT_RATIO
from tst.substrate.adaptive.adaptive_flux_tube import _FACE_HOP, _FACE_NORM

CELLS_ACROSS = 220  # spheromak's own point requirement -- always 220, fixed

R_SPH_CENTRE = np.array([0.0, 0.0, 0.0])
B0 = 1.0
R_SPH = 1.0
K_ = 4.493 / R_SPH
U_MAX, AMPLITUDE = 1.0, 1.0
BATCH = 500_000


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def _B(pos, scale):
    p = (pos - R_SPH_CENTRE) / scale
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    r = np.sqrt(x**2 + y**2 + z**2)
    B = np.zeros_like(p)
    mask = (r <= R_SPH) & (r > 0)
    rm = r[mask]
    cf = B0 * spherical_jn(1, K_ * rm) / rm
    B[mask, 0] = cf * (z[mask] * x[mask] / rm - y[mask])
    B[mask, 1] = cf * (z[mask] * y[mask] / rm + x[mask])
    B[mask, 2] = cf * (-(x[mask] ** 2 + y[mask] ** 2) / rm)
    return B


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--level", type=int, required=True, help="uniform cell-size level (0=coarsest)"
    )
    ap.add_argument("--r-domain", type=float, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cell = 1.0 / SPLIT_RATIO**args.level

    t0 = time.time()
    alat = _lat.AdaptiveLattice()
    alat.add_cluster()
    alat.set_n_threads(8)
    cl = alat[0]
    _lat.build_bcc_sphere(cl, args.r_domain, d1=cell, u0=0.0, ell0=1.0, lam0=0)
    alat.rebuild(0)
    N = cl.n_leaves
    print(
        f"[uniform L{args.level} base] n_leaves={N}  t={time.time()-t0:.1f}s  peak_rss_GB={rss_gb():.2f}",
        flush=True,
    )

    pos = np.array(cl.leaf_pos)
    cell_size = np.full(N, cell)

    SCALE = CELLS_ACROSS / (
        2 * SPLIT_RATIO**args.level
    )  # 240 points at THIS level's own resolution
    print(
        f"cells_across={CELLS_ACROSS}  scale={SCALE:.6f}  (spheromak radius in L{args.level}-cells: {SCALE/cell:.3f})",
        flush=True,
    )

    B_c = _B(pos, SCALE)
    b_mag = np.linalg.norm(B_c, axis=1)
    b_max = float(b_mag.max()) or 1.0
    u_all = U_MAX * b_mag / b_max
    cl.set_leaf_u(u_all)

    r_cell = np.linalg.norm((pos - R_SPH_CENTRE) / SCALE, axis=1)
    n_inner = int((r_cell <= R_SPH).sum())
    print(f"inner_cells={n_inner}/{N} ({100*n_inner/N:.4f}%)", flush=True)

    phi_max = 0.0
    for start in range(0, N, BATCH):
        end = min(start + BATCH, N)
        hop_off = _FACE_HOP[None, :, :] * cell_size[start:end, None, None]
        midpoints = pos[start:end, None, :] + 0.5 * hop_off
        B_mid = _B(midpoints.reshape(-1, 3), SCALE).reshape(end - start, 14, 3)
        phi_batch = np.einsum("jkd,kd->kj", B_mid, _FACE_NORM)
        phi_max = max(phi_max, float(np.abs(phi_batch).max()))
    phi_max = phi_max or 1.0

    phi_full = np.zeros((14, N), dtype=np.float64)
    for start in range(0, N, BATCH):
        end = min(start + BATCH, N)
        hop_off = _FACE_HOP[None, :, :] * cell_size[start:end, None, None]
        midpoints = pos[start:end, None, :] + 0.5 * hop_off
        B_mid = _B(midpoints.reshape(-1, 3), SCALE).reshape(end - start, 14, 3)
        phi_batch = np.einsum("jkd,kd->kj", B_mid, _FACE_NORM)
        phi_batch *= AMPLITUDE / phi_max
        nbr_r = np.linalg.norm(
            (pos[start:end, None, :] + hop_off - R_SPH_CENTRE) / SCALE, axis=2
        )
        phi_batch[(nbr_r > R_SPH).T] = 0.0
        phi_batch[:, r_cell[start:end] > R_SPH] = 0.0
        phi_full[:, start:end] = phi_batch

    cl.set_leaf_phi(np.ascontiguousarray(phi_full))

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
