import sys
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
snap = Path(sys.argv[1]); pos = np.load(snap / "pos.npy")
for t in [13000, 14000, 15000, 16000, 17000, 18000]:
    ck = np.load(snap / f"state_t{t:05d}.npz")
    rho = np.sum(ck["L"] ** 2, 1) + np.sum(ck["R"] ** 2, 1); del ck
    P = pos[rho < 0.25]
    pr = cKDTree(P).query_pairs(1.05, output_type="ndarray"); n = len(P)
    nc, lab = connected_components(coo_matrix((np.ones(len(pr)), (pr[:, 0], pr[:, 1])), shape=(n, n)), directed=False)
    main = np.argmax(np.bincount(lab))
    zc = np.median(P[lab == main][:, 2])
    az = np.degrees(np.arctan2(P[:, 1], P[:, 0])); rr = np.hypot(P[:, 0], P[:, 1])
    out = []
    for k in np.unique(lab):
        Q = (lab == k) & (rr > 100) & (az > 0) & (az < 180)
        if Q.sum() < 50: continue
        a = az[Q]; zz = P[Q, 2] - zc
        out.append(f"{'MAIN' if k == main else 'ring'} n{Q.sum()} az {a.min():.0f}..{a.max():.0f} z-zc {zz.min():+.0f}..{zz.max():+.0f}")
    # closest approach main <-> each ring in the upper half
    Mt = cKDTree(P[lab == main])
    gaps = [f"{Mt.query(P[(lab == k) & (P[:,1] > 0)])[0].min():.1f}" for k in np.unique(lab) if k != main and ((lab == k) & (P[:,1] > 0)).sum() > 200]
    print(f"t {t}: r>100, y>0 -> " + "; ".join(out) + f" | gaps main->rings {gaps}", flush=True)
