import sys

sys.path.insert(0, ".")
import resource

import numpy as np
from scipy.special import spherical_jn

from equilibrium import lattice as _lat
from run.substrate.adaptive.refined_neutron_decay import SPLIT_RATIO
from tst.substrate.adaptive import refine_region_to_convergence
from tst.substrate.adaptive.adaptive_flux_tube import _FACE_HOP, _FACE_NORM

R = 30.0
CORE_LEVEL = 5
CELLS_ACROSS_LIST = [200, 220, 240]
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


# --- build diameter-9 telescope (validated construction) ---
alat = _lat.AdaptiveLattice()
alat.add_cluster()
alat.set_n_threads(8)
cl = alat[0]
_lat.build_bcc_sphere(cl, R, d1=1.0, u0=0.0, ell0=1.0, lam0=0)
alat.rebuild(0)
n0 = cl.n_leaves

pos0 = np.array(cl.leaf_pos)
r0 = np.linalg.norm(pos0, axis=1)
start_j = int(np.argmin(r0))
leaf_nb_g = cl.leaf_nb
leaf_indices = np.array(cl.leaf_global_indices)
max_g = int(leaf_indices.max())
g_to_j = -np.ones(max_g + 1, dtype=np.int64)
g_to_j[leaf_indices] = np.arange(n0)
nb_g_clipped = np.where(leaf_nb_g >= 0, leaf_nb_g, 0)
nb_j0 = np.where(leaf_nb_g >= 0, g_to_j[nb_g_clipped], -1)

hop = np.full(n0, -1, dtype=np.int64)
hop[start_j] = 0
frontier = np.array([start_j])
for d in range(1, CORE_LEVEL + 1):
    nbrs = nb_j0[frontier].ravel()
    nbrs = nbrs[nbrs >= 0]
    nbrs = nbrs[hop[nbrs] == -1]
    nbrs = np.unique(nbrs)
    if len(nbrs) == 0:
        break
    hop[nbrs] = d
    frontier = nbrs

orig_g_at_hop = {d: leaf_indices[hop == d] for d in range(CORE_LEVEL + 1)}

cl.info.split_hi = 0.5
cl.info.split_lo = 0.0

for target_level in range(1, CORE_LEVEL + 1):
    d = CORE_LEVEL - target_level
    target_roots = orig_g_at_hop[d]

    def region_fn(cl, target_roots=target_roots):
        roots = np.array(cl.leaf_ancestry) >> 32
        return np.isin(roots, target_roots)

    refine_region_to_convergence(alat, cl, region_fn, stage_max_level=target_level)
    print(
        f"  [telescope stage L{target_level}] n_leaves={cl.n_leaves}  peak_rss_GB={rss_gb():.2f}",
        flush=True,
    )

N = cl.n_leaves
print(f"[telescope built] n_leaves={N}  peak_rss_GB={rss_gb():.2f}", flush=True)

pos = np.array(cl.leaf_pos)
levels = np.array(cl.leaf_level)
cell_size = 1.0 / (SPLIT_RATIO**levels)

for CELLS_ACROSS in CELLS_ACROSS_LIST:
    SCALE = CELLS_ACROSS / (2 * SPLIT_RATIO**CORE_LEVEL)
    print(f"\n=== cells_across={CELLS_ACROSS}  scale={SCALE:.6f} ===", flush=True)

    B_c = _B(pos, SCALE)
    b_mag = np.linalg.norm(B_c, axis=1)
    b_max = float(b_mag.max()) or 1.0
    u_all = U_MAX * b_mag / b_max

    # Buffer check: how much field magnitude remains in the outermost few
    # L5-cell-widths before the L5/L4 boundary at r0=0.5?
    r0 = np.linalg.norm(pos, axis=1)
    l5_w = 1.0 / SPLIT_RATIO**CORE_LEVEL
    for n_cells_buf in (1, 2, 3, 5):
        buf_lo = 0.5 - n_cells_buf * l5_w
        in_buf = (r0 >= buf_lo) & (r0 < 0.5)
        buf_max = float(b_mag[in_buf].max()) if in_buf.any() else 0.0
        print(
            f"  buffer[{n_cells_buf} cells before r=0.5]: n={int(in_buf.sum())}  "
            f"max|B|={buf_max:.4e}  (vs overall max|B|={b_max:.4e}, ratio={buf_max/b_max:.4e})"
        )
    cl.set_leaf_u(u_all)

    r_cell = np.linalg.norm((pos - R_SPH_CENTRE) / SCALE, axis=1)
    n_inner = int((r_cell <= R_SPH).sum())
    print(f"inner_cells={n_inner}/{N} ({100*n_inner/N:.4f}%)")

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

print(f"\nFINAL peak_rss_GB={rss_gb():.2f}")
