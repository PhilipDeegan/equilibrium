import sys

sys.path.insert(0, ".")
import resource
import time

import numpy as np
from scipy.special import spherical_jn

from equilibrium import lattice as _lat
from run.substrate.adaptive.refined_neutron_decay import SPLIT_RATIO
from tst.substrate.adaptive import refine_region_to_convergence
from tst.substrate.adaptive.adaptive_flux_tube import _FACE_HOP, _FACE_NORM

R = 30.0
CORE_LEVEL = 5
CELLS_ACROSS = 220
# real-distance thresholds per level (isotropic, replaces old hop-count cube
# construction). Original L4/L3/L2/L1 thresholds (1.82/2.91/4.01/5.02) were
# copied from the old hop-based construction's best-direction reach, which
# turned out to give wildly inconsistent annulus widths in each level's own
# cell size: L4~108, L3~29, L2~10, L1~3 -- L4 alone was 17.15M of 27.2M total
# leaves. No level's annulus needs to exceed ~22 cells wide (own units); L2/L1
# were already under that, so only L4 and L3 actually get capped down. L5
# (the core, not a shell) keeps its dedicated sizing: the raw cells_across=220
# requirement (37.5 L4-cell-widths of radius) plus 1 L4-cell of margin to
# "de-cube" the L4/L5 transition (that margin band gets unevenly perforated
# into L5 by the real-distance criterion rather than sitting as a flat cutoff).
L4_CELL = 1.0 / SPLIT_RATIO**4
L3_CELL = 1.0 / SPLIT_RATIO**3
L2_CELL = 1.0 / SPLIT_RATIO**2
L1_CELL = 1.0 / SPLIT_RATIO**1
ANNULUS_CAP = 22.0

r5 = (37.5 + 1.0) * L4_CELL  # 37.5 (225/2/3 margin baseline) +1 L4-cell de-cube
r4 = r5 + ANNULUS_CAP * L4_CELL  # L4 annulus capped at 22 L4-cells
r3 = r4 + ANNULUS_CAP * L3_CELL  # L3 annulus capped at 22 L3-cells
r2 = r3 + 10.0 * L2_CELL  # L2 annulus unchanged (~10, already < cap)
r1 = r2 + 3.0 * L1_CELL  # L1 annulus unchanged (~3, already < cap)
THRESHOLDS = {1: r1, 2: r2, 3: r3, 4: r4, 5: r5}
R_SPH_CENTRE = np.array([0.0, 0.0, 0.0])
B0 = 1.0
R_SPH = 1.0
K_ = 4.493 / R_SPH
U_MAX, AMPLITUDE = 1.0, 1.0
BATCH = 500_000
OUT_PATH = "/bk/l5_realdist_core_ic_220.h5"


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


t0 = time.time()
alat = _lat.AdaptiveLattice()
alat.add_cluster()
alat.set_n_threads(8)
cl = alat[0]
_lat.build_bcc_sphere(cl, R, d1=1.0, u0=0.0, ell0=1.0, lam0=0)
alat.rebuild(0)
print(
    f"[base] n_leaves={cl.n_leaves}  t={time.time()-t0:.1f}s  peak_rss_GB={rss_gb():.2f}",
    flush=True,
)

cl.info.split_hi = 0.5
cl.info.split_lo = 0.0

for target_level in range(1, CORE_LEVEL + 1):
    threshold = THRESHOLDS[target_level]

    def region_fn(cl, threshold=threshold, target_level=target_level):
        pos = np.array(cl.leaf_pos)
        levels = np.array(cl.leaf_level)
        dist = np.linalg.norm(pos, axis=1)
        return (levels < target_level) & (dist <= threshold)

    t1 = time.time()
    refine_region_to_convergence(alat, cl, region_fn, stage_max_level=target_level)
    print(
        f"[L{target_level}, r<={threshold:.4f}] n_leaves={cl.n_leaves}  "
        f"t={time.time()-t1:.1f}s  peak_rss_GB={rss_gb():.2f}",
        flush=True,
    )

N = cl.n_leaves
levels = np.array(cl.leaf_level)
print(
    f"[telescope built] n_leaves={N}  level_hist={np.bincount(levels).tolist()}  "
    f"total_t={time.time()-t0:.1f}s  peak_rss_GB={rss_gb():.2f}",
    flush=True,
)

pos = np.array(cl.leaf_pos)
cell_size = 1.0 / (SPLIT_RATIO**levels)

SCALE = CELLS_ACROSS / (2 * SPLIT_RATIO**CORE_LEVEL)
print(f"cells_across={CELLS_ACROSS}  scale={SCALE:.6f}", flush=True)

B_c = _B(pos, SCALE)
b_mag = np.linalg.norm(B_c, axis=1)
b_max = float(b_mag.max()) or 1.0
u_all = U_MAX * b_mag / b_max

r0 = np.linalg.norm(pos, axis=1)
l5_w = 1.0 / SPLIT_RATIO**CORE_LEVEL
for n_cells_buf in (1, 2, 3, 5):
    buf_lo = 0.5 - n_cells_buf * l5_w
    in_buf = (r0 >= buf_lo) & (r0 < 0.5)
    buf_max = float(b_mag[in_buf].max()) if in_buf.any() else 0.0
    print(
        f"  buffer[{n_cells_buf} cells before r=0.5]: n={int(in_buf.sum())}  "
        f"max|B|={buf_max:.4e}  (vs overall max|B|={b_max:.4e}, ratio={buf_max/b_max:.4e})",
        flush=True,
    )
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
alat.save(OUT_PATH)
print(f"saved {OUT_PATH}  save_t={time.time()-t2:.1f}s", flush=True)
print(f"FINAL total_t={time.time()-t0:.1f}s  peak_rss_GB={rss_gb():.2f}", flush=True)
