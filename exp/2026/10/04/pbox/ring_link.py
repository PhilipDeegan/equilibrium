"""do the small rings encircle the main ring? per small ring: centroid, normal, radius; the main ring core cells
within 2 of its plane, their in-plane distance from its centroid (threaded if < ring radius). periodic box.
usage: ring_link.py snap_dir P t [t ...]"""
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

snap, P = Path(sys.argv[1]), int(sys.argv[2])
side = P / np.sqrt(3)
box = np.mod(np.load(snap / "pos.npy") + side / 2, side)
wrap = lambda d: (d + side / 2) % side - side / 2
for t in map(int, sys.argv[3:]):
    ck = np.load(snap / f"state_t{t:05d}.npz")
    rho = np.sum(ck["L"] ** 2, 1) + np.sum(ck["R"] ** 2, 1)
    del ck
    Q = box[rho < 0.25]
    pr = cKDTree(Q, boxsize=side).query_pairs(1.05, output_type="ndarray")
    nc, lab = connected_components(coo_matrix((np.ones(len(pr)), (pr[:, 0], pr[:, 1])), shape=(len(Q),) * 2), directed=False)
    sz = np.bincount(lab)
    order = [k for k in np.argsort(-sz) if sz[k] >= 500]
    main = Q[lab == order[0]]
    cm = (np.angle(np.exp(1j * main / side * 2 * np.pi).mean(0)) / (2 * np.pi) * side) % side
    print(f"t {t}: main {len(main)} cells, centre z {cm[2] - side / 2:+.1f}")
    for k in order[1:]:
        X = Q[lab == k]
        c = (np.angle(np.exp(1j * X / side * 2 * np.pi).mean(0)) / (2 * np.pi) * side) % side
        d = wrap(X - c)
        nrm = np.linalg.svd(d, full_matrices=False)[2][2]
        rad = np.linalg.norm(d - np.outer(d @ nrm, nrm), axis=1).mean()
        dm = wrap(main - c)
        h = dm @ nrm
        near = np.abs(h) < 2
        inpl = np.linalg.norm(dm[near] - np.outer(h[near], nrm), axis=1)
        dzc = wrap(c - cm)
        rho_c = np.hypot(*dzc[:2])
        print(f"  {len(X):5d} cells, rel. main centre (rho {rho_c:5.1f}, dz {dzc[2]:+6.1f}), radius {rad:4.1f}, normal "
              f"{np.round(nrm, 2)} (radial comp {abs(nrm[:2] @ dzc[:2]) / max(rho_c, 1e-9):.2f}); main crosses plane at "
              f"in-plane dist min {inpl.min() if near.any() else np.nan:5.1f} -> "
              f"{'THREADED' if near.any() and inpl.min() < rad else 'not threaded'}", flush=True)
