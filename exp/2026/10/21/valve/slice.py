"""10-21 valve: where the tilted pair's circulation sits -- |Gamma| max-projections (x-z, x-y) and the midplane (y-z) slab."""
import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from equilibrium.substrate import continuous as C
from equilibrium.substrate.walk import cpp_box

p = argparse.ArgumentParser()
p.add_argument("files", nargs="+")
p.add_argument("--P", type=int, default=192)
p.add_argument("--R", type=float, default=20.0)
p.add_argument("--out", default="exp/2026/10/21/valve/slice.png")
args = p.parse_args()
side = args.P / np.sqrt(3)
D, W = C.D_HAT, C.W
OFF = np.array([0.13, 0.07, 0.11])
wrapx = lambda x: (x + side / 2) % side - side / 2
ac = C.AnchorCluster(cpp_box(args.P, n_threads=8))
pos, nb = ac.pos, ac.nb
tri = [(ka, kb, ks) for ka in range(8) for kb in range(8) for ks in range(8, 14) if np.allclose(D[kb] - D[ka], D[ks])]
cen = wrapx(np.stack([pos + (D[ka] + D[kb]) / 3 for ka, kb, _ in tri], 1).reshape(-1, 3) - OFF)
bins = np.linspace(-side / 2, side / 2, int(side) + 1)

fig, axs = plt.subplots(len(args.files), 3, figsize=(15, 4.6 * len(args.files)), squeeze=False)
for row, f in zip(axs, args.files):
    a = np.load(f)["phi"] / W
    g = np.abs(np.stack([a[:, ka] + a[nb[:, ka], ks] - a[:, kb] for ka, kb, ks in tri], 1).ravel()) / (2 * np.pi)
    for ax, (i, j), m, lab in zip(row, [(0, 2), (0, 1), (1, 2)], [None, None, np.abs(cen[:, 0]) < args.R / 8],
                                  ["x-z (max over y)", "x-y (max over z)", "y-z, |x| < R / 8 (max)"]):
        s = slice(None) if m is None else m
        H = np.zeros((len(bins) - 1,) * 2)
        ix, iy = np.digitize(cen[s, i], bins) - 1, np.digitize(cen[s, j], bins) - 1
        np.maximum.at(H, (ix.clip(0, len(bins) - 2), iy.clip(0, len(bins) - 2)), g[s])
        im = ax.imshow(H.T, origin="lower", extent=[bins[0], bins[-1]] * 2, cmap="magma")
        ax.set_title(f"{Path(f).stem}: {lab}  max {H.max():.3f}")
        fig.colorbar(im, ax=ax, shrink=0.8)
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
fig.tight_layout()
fig.savefig(args.out, dpi=110)
print(args.out)
