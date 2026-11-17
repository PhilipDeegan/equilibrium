"""connected core pieces in a periodic box state (periodic adjacency), with size, centroid (periodic mean), extent,
and tube winding around a small loop at a few core points (amp = pair angle of the k = 0 branch)."""
import sys
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from equilibrium.substrate.walk import rest_state, to_complex

snap, P = Path(sys.argv[1]), int(sys.argv[2])
ts = [int(a) for a in sys.argv[3:]]
side = P / np.sqrt(3)
pos = np.load(snap / "pos.npy")
v, _ = rest_state(0.8)
box = np.mod(pos + side / 2, side)
tree_all = cKDTree(box, boxsize=side)
for t in ts:
    ck = np.load(snap / f"state_t{t:05d}.npz")
    rho = np.sum(ck["L"] ** 2, 1) + np.sum(ck["R"] ** 2, 1)
    amp = np.concatenate([to_complex(ck["L"]), to_complex(ck["R"])], 1) @ v.conj()
    del ck
    idx = np.flatnonzero(rho < 0.25)
    Q = box[idx]
    pr = cKDTree(Q, boxsize=side).query_pairs(1.05, output_type="ndarray")
    nc, lab = connected_components(coo_matrix((np.ones(len(pr)), (pr[:, 0], pr[:, 1])), shape=(len(Q),) * 2), directed=False)
    sz = np.bincount(lab)
    print(f"t {t}: {nc} pieces, sizes {sorted(sz.tolist(), reverse=True)[:12]}; background rho sd {rho[rho > 0.5].std():.3f}")
    for k in np.argsort(-sz)[:12]:
        if sz[k] < 100:
            continue
        X = Q[lab == k]
        ang = X / side * 2 * np.pi
        c = (np.angle(np.exp(1j * ang).mean(0)) / (2 * np.pi) * side) % side
        d = (X - c + side / 2) % side - side / 2
        _, s, vt = np.linalg.svd(d, full_matrices=False)
        nrm = vt[2]
        wnd = []
        for j in np.random.default_rng(0).choice(len(X), 4, replace=False):
            p = X[j]
            e = d[j] - (d[j] @ nrm) * nrm
            e /= np.linalg.norm(e) + 1e-12
            ph = np.linspace(0, 2 * np.pi, 96, endpoint=False)
            loop = p[None] + 5 * (np.cos(ph)[:, None] * e[None] + np.sin(ph)[:, None] * nrm[None])
            a = amp[tree_all.query(np.mod(loop, side))[1]]
            wnd.append(np.sum(np.angle(a[np.r_[1:len(a), 0]] / a)) / (2 * np.pi))
        cc = c - side / 2
        print(f"   {sz[k]:6d} at ({cc[0]:+6.1f},{cc[1]:+6.1f},{cc[2]:+6.1f}) extents sd {np.round(s / np.sqrt(len(X)), 1)}; "
              f"winding " + " ".join(f"{w:+.0f}" for w in wnd), flush=True)
