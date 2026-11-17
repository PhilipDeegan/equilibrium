"""phase winding (pair angle of the k = 0 branch) around the tube of each detached core piece: small loop of radius
rl around a core point of the piece, in the plane (e, n) (e = in-plane radial of the ring, n = ring normal, outward-
radial sign); right-handed about t = n x e. Also the main strand's winding around its own tube at one point."""
import sys
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from equilibrium.substrate.walk import rest_state, to_complex

snap, m, rl = Path(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
ts = [int(a) for a in sys.argv[4:]]
v, _ = rest_state(m)
pos = np.load(snap / "pos.npy")
tree = cKDTree(pos)
for t in ts:
    ck = np.load(snap / f"state_t{t:05d}.npz")
    L, R = ck["L"], ck["R"]
    rho = np.sum(L**2, 1) + np.sum(R**2, 1)
    amp = np.concatenate([to_complex(L), to_complex(R)], 1) @ v.conj()
    del ck, L, R
    idx = np.flatnonzero(rho < 0.25)
    P = pos[idx]
    pr = cKDTree(P).query_pairs(1.05, output_type="ndarray")
    nc, lab = connected_components(coo_matrix((np.ones(len(pr)), (pr[:, 0], pr[:, 1])), shape=(len(P),) * 2), directed=False)
    sizes = np.bincount(lab)
    print(f"t {t}: pieces {sorted(sizes.tolist(), reverse=True)[:10]}")
    for k in np.argsort(-sizes):
        if sizes[k] < 200:
            continue
        Q = P[lab == k]
        c = Q.mean(0)
        _, _, vt = np.linalg.svd(Q - c, full_matrices=False)
        nrm = vt[2] * np.sign(vt[2] @ c)
        out = []
        for ang in np.linspace(0, 2 * np.pi, 4, endpoint=False):  # 4 points around the ring
            e0 = vt[0] * np.cos(ang) + vt[1] * np.sin(ang)
            j = np.argmin(np.linalg.norm(Q - c, axis=1) - 2 * (Q - c) @ e0)  # core cell far along e0
            p = Q[j]
            e = p - c - ((p - c) @ nrm) * nrm
            e /= np.linalg.norm(e)
            ph = np.linspace(0, 2 * np.pi, 128, endpoint=False)
            loop = p[None] + rl * (np.cos(ph)[:, None] * e[None] + np.sin(ph)[:, None] * nrm[None])
            a = amp[tree.query(loop)[1]]
            out.append(np.sum(np.angle(a[np.r_[1:len(a), 0]] / a)) / (2 * np.pi))
        tag = "MAIN " if sizes[k] == sizes.max() else "piece"
        print(f"   {tag} {sizes[k]:6d} at ({c[0]:+6.1f},{c[1]:+6.1f},{c[2]:+6.1f}): tube winding "
              + " ".join(f"{w:+.2f}" for w in out), flush=True)
