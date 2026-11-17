"""closest self-approach of the main core strand: core cells (rho < 0.25) coarse-grained to voxels (edge V), voxel
adjacency graph, geodesic distance along the core; neck = min euclidean distance between voxels whose geodesic
separation is >= G (parts of the strand far apart along it). Reported per state with the cylindrical radius and
azimuth of the closest pair."""
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, shortest_path
from scipy.spatial import cKDTree

snap = Path(sys.argv[1]); V = 3.0; G = float(sys.argv[2]); ts = [int(a) for a in sys.argv[3:]]
pos = np.load(snap / "pos.npy")
for t in ts:
    ck = np.load(snap / f"state_t{t:05d}.npz")
    rho = np.sum(ck["L"] ** 2, 1) + np.sum(ck["R"] ** 2, 1); del ck
    P = pos[rho < 0.25]
    key, inv = np.unique(np.floor(P / V).astype(int), axis=0, return_inverse=True)
    inv = inv.ravel()
    C = np.stack([np.bincount(inv, P[:, k]) for k in range(3)], 1) / np.bincount(inv)[:, None]
    pr = cKDTree(key).query_pairs(np.sqrt(3) + 1e-6, output_type="ndarray")
    n = len(key)
    d = np.linalg.norm(C[pr[:, 0]] - C[pr[:, 1]], axis=1)
    A = coo_matrix((d, (pr[:, 0], pr[:, 1])), shape=(n, n)).tocsr()
    nc, lab = connected_components(A, directed=False)
    big = np.flatnonzero(lab == np.argmax(np.bincount(lab)))
    Ab = A[big][:, big]
    D = shortest_path(Ab, directed=False)
    Cb = C[big]
    tr = cKDTree(Cb)
    best = (np.inf, None)
    for i, j in tr.query_pairs(60.0, output_type="ndarray"):
        if D[i, j] >= G:
            e = np.linalg.norm(Cb[i] - Cb[j])
            if e < best[0]:
                best = (e, (i, j))
    e, (i, j) = best
    m = 0.5 * (Cb[i] + Cb[j])
    print(f"t {t:5d}: main voxels {len(big)} (pieces {nc}); neck {e:5.1f} at rho_cyl {np.hypot(*m[:2]):5.1f} "
          f"az {np.degrees(np.arctan2(m[1], m[0])):+6.0f}, geodesic {D[i, j]:.0f}", flush=True)
