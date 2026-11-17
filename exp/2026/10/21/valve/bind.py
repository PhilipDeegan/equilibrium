"""10-21: do the 10-19 R 20 equal ring pairs bind? per-half |Gamma|-weighted x centroid, seed vs end."""
import sys

import numpy as np

from equilibrium.substrate import continuous as C
from equilibrium.substrate.walk import cpp_box

P = 192
side = P / np.sqrt(3)
D, W = C.D_HAT, C.W
OFF = np.array([0.13, 0.07, 0.11])
wrapx = lambda x: (x + side / 2) % side - side / 2
ac = C.AnchorCluster(cpp_box(P, n_threads=8))
pos, nb = ac.pos, ac.nb
tri = [(ka, kb, ks) for ka in range(8) for kb in range(8) for ks in range(8, 14) if np.allclose(D[kb] - D[ka], D[ks])]
cen = wrapx(np.stack([pos + (D[ka] + D[kb]) / 3 for ka, kb, _ in tri], 1).reshape(-1, 3) - OFF)
for name in sys.argv[1:]:
    out = []
    for f in ("seed", "end"):
        a = np.load(f"/bk/dst/2026/10/19/recv/{f}_{name}.npz")["phi"] / W
        g = np.abs(np.stack([a[:, ka] + a[nb[:, ka], ks] - a[:, kb] for ka, kb, ks in tri], 1).ravel())
        xl = np.sum(g * (cen[:, 0] < 0) * cen[:, 0]) / np.sum(g * (cen[:, 0] < 0))
        xr = np.sum(g * (cen[:, 0] >= 0) * cen[:, 0]) / np.sum(g * (cen[:, 0] >= 0))
        out.append((xl, xr, xr - xl))
    print(f"{name}: seed x {out[0][0]:+.3f} {out[0][1]:+.3f} sep {out[0][2]:.3f}  end x {out[1][0]:+.3f} {out[1][1]:+.3f} sep {out[1][2]:.3f}  d sep {out[1][2] - out[0][2]:+.4f}", flush=True)
