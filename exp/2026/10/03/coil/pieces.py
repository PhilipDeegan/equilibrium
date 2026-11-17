"""connected pieces of the core (rho < 0.25) in saved R175 states: size, centroid, flatness, ring radius, hole,
normal, distance to the main strand; pieces matched across times by nearest centroid."""
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

snap = Path(sys.argv[1])
ts = [int(a) for a in sys.argv[2:]]
pos = np.load(snap / "pos.npy")
r_all = np.linalg.norm(pos, axis=1)
for t in ts:
    ck = np.load(snap / f"state_t{t:05d}.npz")
    rho = np.sum(ck["L"] ** 2, 1) + np.sum(ck["R"] ** 2, 1)
    zs = float(ck["z_shift"])
    del ck
    idx = np.flatnonzero(rho < 0.25)
    P = pos[idx]
    pr = cKDTree(P).query_pairs(1.05, output_type="ndarray")
    n = len(P)
    nc, lab = connected_components(coo_matrix((np.ones(len(pr)), (pr[:, 0], pr[:, 1])), shape=(n, n)), directed=False)
    sizes = np.bincount(lab)
    order = np.argsort(-sizes)
    main = P[lab == order[0]]
    tm = cKDTree(main)
    print(f"t {t} (z_shift {zs:+.1f}): {nc} pieces, sizes {sizes[order][:12].tolist()}")
    for k in order[1:]:
        if sizes[k] < 200:
            continue
        Q = P[lab == k]
        c = Q.mean(0)
        u, s, vt = np.linalg.svd(Q - c, full_matrices=False)
        sd = s / np.sqrt(len(Q))
        nrm = vt[2] * np.sign(vt[2] @ c if abs(vt[2] @ c) > 1e-9 else 1)
        inpl = (Q - c) - np.outer((Q - c) @ vt[2], vt[2])
        rad = np.linalg.norm(inpl, axis=1).mean()
        hole = cKDTree(Q).query(c)[0]
        dmain = tm.query(Q)[0].min()
        rc = np.linalg.norm(c)
        radial = c / rc
        print(f"   piece {sizes[k]:6d}: centroid ({c[0]:+6.1f},{c[1]:+6.1f},{c[2]:+6.1f}) |c| {rc:5.1f} "
              f"r max {np.linalg.norm(Q, axis=1).max():5.1f}; sd across {sd[2]:.1f}, ring radius {rad:.1f}, "
              f"hole {hole:.1f}; normal.radial {nrm @ radial:+.2f} normal_z {nrm[2]:+.2f}; gap to main {dmain:.1f}")
    print(f"   main strand r max {np.linalg.norm(main, axis=1).max():.1f}", flush=True)
